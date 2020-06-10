import org.apache.spark.{SparkConf,SparkContext}

object A2Q1PageRank {

  def main(args: Array[String]) {
    if (args.length != 3) {
      println("Require Input Path, Output Path and number of Iterations")
    }
    val pathForInput = args(0)
    val pathForOutput = args(1)
    val totalIterations: Int = args(2).toInt

    val alpha = 0.15
    val sc = new SparkContext(new SparkConf().setAppName("A2Q1PageRank"))
    val inputData = sc.textFile(pathForInput)
    val inputHeader = inputData.first()
    val inputLinks = inputData.filter(r => r != inputHeader).map(eachRow => {
      val inputPairs = eachRow.split(",")
      (inputPairs(0),inputPairs(1))
    }).groupByKey()

    val inputTotal = inputLinks.count()
    var inputRanks = inputLinks.mapValues(v => 10.0)

    for (i <- 1 to totalIterations) {
      val airportsFromRank = inputLinks.join(inputRanks).values.flatMap(pageRanksToN => {
        val rankOfFromAirport = pageRanksToN._2
        val linksOut = pageRanksToN._1.size
        val listOfToAirports = pageRanksToN._1

        listOfToAirports.map(pageTO => {
          val rankOfFromAirports = rankOfFromAirport / linksOut;
          (pageTO, rankOfFromAirports)
        })
      })

      inputRanks = airportsFromRank.reduceByKey(_ + _).mapValues(prs => ((1 - alpha)*prs+alpha) / inputTotal)
    }
    inputRanks = inputRanks.sortBy(_._2, false)

    val outputResult = inputRanks.map(eachAirportRank => eachAirportRank._1 + "\t" + eachAirportRank._2)
    outputResult.coalesce(1).saveAsTextFile(pathForOutput)
  }
}