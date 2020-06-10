import org.apache.spark.ml.feature.{HashingTF, StopWordsRemover, StringIndexer, Tokenizer}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession


object TweetProc {
  def main(args: Array[String]): Unit = {

    val sparkSession = SparkSession.builder.appName("Tweet Sentiment").getOrCreate()
    val sparkCont = sparkSession.sparkContext

    import sparkSession.implicits._

    val inputFile=args(0)
    val outputFile=args(1)

    val dfInput = sparkSession.read.option("header","true").option("inferSchema","true").csv(inputFile).toDF().select("text","airline_sentiment").toDF().na.drop()
    val tweetTokenizer = new Tokenizer().setInputCol("text").setOutputCol("textWords")
    val removerOfStopWords = new StopWordsRemover().setInputCol(tweetTokenizer.getOutputCol).setOutputCol("filteredTextWords")
    val hasher = new HashingTF().setInputCol(removerOfStopWords.getOutputCol).setOutputCol("features")
    val stringIndex = new StringIndexer().setInputCol("airline_sentiment").setOutputCol("label").fit(dfInput)
    val tweetValues = sparkSession.createDataFrame(Seq((0, "neutral"),(1, "negative"), (2, "positive"))).toDF("id", "airline_sentiment")
    val tweetMap = stringIndex.transform(tweetValues).select("airline_sentiment", "label")

    val logisticRegression = new LogisticRegression()
    val paramGrid = new ParamGridBuilder()
      .addGrid(logisticRegression.elasticNetParam,Array(0.7,0.8,0.9))
      .addGrid(logisticRegression.regParam, Array(0.1, 0.01))
      .addGrid(hasher.numFeatures, Array(10,100,1000)).build()

    val pip = new Pipeline().setStages(Array(tweetTokenizer, removerOfStopWords, hasher, stringIndex, logisticRegression))
    val Array(training, test) = dfInput.randomSplit(Array(0.8, 0.2), seed = 123)

    val crossVal = new CrossValidator()
      .setEstimator(pip)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(8)
      .setParallelism(18)

    val logisticRegModel = crossVal.fit(training)
    val bestLRModel=logisticRegModel.bestModel
    val freshModel=bestLRModel.transform(test)
      .select(hasher.getOutputCol,"label","prediction")

    val predictAndLabels = freshModel.select("label", "prediction")
      .map(x => (x.getDouble(0), x.getDouble(1))).rdd

    val evaluationM = new MulticlassMetrics(predictAndLabels)
    var stringBuilder = new StringBuilder()
    stringBuilder.append("")

    stringBuilder.append("\nCONFUSION MATRIX:\n")
    stringBuilder.append("\n"+evaluationM.confusionMatrix)

    stringBuilder.append("\nSENTIMENT TWEET VALUES:\n")
    tweetMap.collect().foreach(i=>stringBuilder.append("%s = %f\n".format(i.getString(0),i.getDouble(1))))

    val modelAccuracy = evaluationM.accuracy
    val modelLabels = evaluationM.labels

    stringBuilder.append("\nSTATISTICS:\n")

    stringBuilder.append("\tModelPrecision by label")
    modelLabels.foreach { row =>
      stringBuilder.append(s"ModelPrecision($row) = " + evaluationM.precision(row))
    }

    stringBuilder.append("\tModelRecall by label")
    modelLabels.foreach { row =>
      stringBuilder.append(s"ModelRecall($row) = " + evaluationM.recall(row))
    }

    stringBuilder.append("\tF1Score by label")
    modelLabels.foreach { row =>
      stringBuilder.append(s"F1Score($row) = " + evaluationM.fMeasure(row))
    }

    stringBuilder.append("\tFalsePositive by label")
    modelLabels.foreach { row =>
      stringBuilder.append(s"FalsePositive($row) = " + evaluationM.falsePositiveRate(row))
    }

    stringBuilder.append("\n\nSUMMARY:\n")
    stringBuilder.append(s"\nAccuracy = $modelAccuracy")
    stringBuilder.append(s"\nAverage Model Precision: ${evaluationM.weightedPrecision}")
    stringBuilder.append(s"\nAverage Model Recall: ${evaluationM.weightedRecall}")
    stringBuilder.append(s"\nAverage F1Score: ${evaluationM.weightedFMeasure}")
    stringBuilder.append(s"\nAverage False positive rate: ${evaluationM.weightedFalsePositiveRate}")

    sparkCont.parallelize(Seq(stringBuilder.toString())).coalesce(1).saveAsTextFile(outputFile)

  }
}