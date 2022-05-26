package com.tt.ie.dm.activity
import com.tt.ie.dm.utils.{SparkConfiguration, Client}
import org.apache.spark.mllib.util.tt.OptionParser
import org.apache.spark.sql.DataFrame

import scala.math
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{avg, col, log, pow, row_number, sum}


object NDCGp{
  def main(args: Array[String]): Unit = {
    // 初始化spark环境
    val options = new OptionParser(args)
    val spark = SparkConfiguration.getSparkSession()
    val client = new Client()
    //配置输入的参数
    val p = options.getString("p", "10").toInt //表示计算NDCGp指标中的p值

    import spark.implicits._
    // 输入的信息表格
    val Array(userDatabase, userTable) = options.getString("data_input").split(",")(0).split("::")
    val table: DataFrame = client.tdw(userDatabase)
      .table(userTable)
      .map(r => (r(0), r(1), r(2), r(3), r(4))) //roleid,clubid,cosinesim,distance,ranking
      .toDF(Seq("roleid", "clubid", "cosinesim", "distance", "ranking"): _*)

    //先计算每一个对象的DCG指标
    val tableDCG = table.withColumn("dcgs", (pow(2, col("cosinesim").cast("double")) - 1) / (log(col("ranking").cast("double") + 1) / scala.math.log(2.0)))
      .where(col("ranking") <= p)
      .groupBy("roleid")
      .agg(sum("dcgs"))
      .withColumnRenamed("sum('dcgs')", "dcg")

    val w = Window.partitionBy(col("roleid")).orderBy(col("cosinesim").desc)

    val tableIDCG = table.select("roleid", "clubid", "cosinesim", "distance")
      .withColumn("ranking", row_number.over(w))
      .withColumn("idcgs", (pow(2, col("cosinesim")) - 1) / (log(col("ranking").cast("double") + 1) / scala.math.log(2.0)))
      .where(col("ranking") <= p)
      .groupBy("roleid")
      .agg(sum("idcgs"))
      .withColumnRenamed("sum('idcgs')", "idcg")

    //将DCG与IDCG表格拼接起来
    val resultDf = tableDCG.join(tableIDCG, tableDCG("roleid") === tableIDCG("roleid"), "left_outer")
      .toDF(Seq("roleid", "dcg", "roleid2", "idcg"): _*)
      .where(col("roleid2").isNotNull)
      .select("roleid", "dcg", "idcg")
      .withColumn("ndgc", col("dcg").cast("double") / col("idcg").cast("double"))
      .agg(avg("dcg"),avg("dcg"),avg("ndgc"))
      .withColumnRenamed("avg('dcg')","dcg")
      .withColumnRenamed("avg('idcg')","idcg")
      .withColumnRenamed("avg('ndcg')","ndcg")

    resultDf.show()

    // 输出结果,将表格的数据写入分布式数据库中
    val Array(resultdb, resulttable) =
      options.getString("data_output").split(",")(0).split("::")
    client.writeDataFrameToTable(
      resultdb,
      resulttable,
      resultDf,
      partValue = "",
      replace = false
    )
  }
}
