package kiran

import java.nio.charset.CodingErrorAction

import breeze.numerics.sqrt
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

import scala.io.{Codec, Source}

object MovieSimilarities {

  type ratingPair = (Double, Double)
  type ratingPairs = Iterable[ratingPair]
  def computeCosineSimilarity(userRatingPairs: ratingPairs): (Double, Int) ={
    var numPairs: Int = 0
    var sum_xx:Double = 0.0
    var sum_yy:Double = 0.0
    var sum_xy:Double = 0.0

    for(pair <- userRatingPairs){
      val ratingX = pair._1
      val ratingY = pair._2

      sum_xx += ratingX * ratingX
      sum_yy += ratingY * ratingY
      sum_xy += ratingX * ratingY
      numPairs+=1
    }

    val numerator:Double = sum_xy
    val denominator = sqrt(sum_xx) + sqrt(sum_yy)

    var score:Double =0.0

    if(denominator!=0){
      score = numerator/denominator
    }

    return (score,numPairs)
  }

  def loadMovieData():Map[Int,String]={

    //To handle encoding issues
    implicit val codec = Codec("UTF-8")
    codec.onMalformedInput(CodingErrorAction.REPLACE)
    codec.onUnmappableCharacter(CodingErrorAction.REPLACE)


    var movieData:Map[Int, String] = Map()

    val data = Source.fromFile("./data/u.item").getLines()

    for(line <- data){
      val fields = line.split('|')

      if(fields.length>1){
        movieData += (fields(0).toInt -> fields(1).toString)
      }
    }

    return movieData
  }

  type movieRating = (Int, Double)
  type userRatingPair = (Int, (movieRating, movieRating))

  def makePairs(userRatings: userRatingPair): ((Int, Int), (Double, Double)) ={
    val movieId1 = userRatings._2._1._1
    val rating1 = userRatings._2._1._2

    val movieId2 = userRatings._2._2._1
    val rating2 = userRatings._2._2._2

    return ((movieId1,movieId2),(rating1,rating2))
  }
  def userIdMovieIdRatingFunction(line:String): (Int, (Int, Double)) ={
    val lineField = line.split("\t")

    //println(lineField(0))

    return (lineField(0).toInt, (lineField(1).toInt, lineField(2).toDouble))
  }

  def filterDuplicateMovies(userRatings: userRatingPair):Boolean ={
    val movieId1 = userRatings._2._1._1
    val movieId2 = userRatings._2._2._1

    return movieId1 < movieId2
  }

  def main(args:Array[String]): Unit ={
    Logger.getLogger("org").setLevel(Level.ERROR)

    val sc = new SparkContext("local[*]","MovieRecommendation")
    //Create movieId -> Name dictionary
    val movieNamedDict = loadMovieData()

    //Create a data of user id -> (movieid, rating) for each row
    val userIdMovieIdRating = sc.textFile("./data/u.data").map(userIdMovieIdRatingFunction)


    //Get every combination of movies which the user has watched together
    val userIdMovieIdRatingJoin = userIdMovieIdRating.join(userIdMovieIdRating)

    val filteredMovie = userIdMovieIdRatingJoin.filter(filterDuplicateMovies)


    val moviePairs = filteredMovie.map(makePairs)

    //For a given pair of movie collect all the movie pairs

    val moviePairRatings = moviePairs.groupByKey()


    //This will have the similarities between 2 movies based on the cosine similarity function
    val moviePairSimilarities = moviePairRatings.mapValues(computeCosineSimilarity).cache()

    if(args.length>0){
      val scoreThreshold = 0.97
      val coOcurrenceThreshold = 50.0

      val movieId:Int = args(0).toInt


      //Get movies which are having a score of greater than score threshold and has occurred at leas the coOccurenceThreshold

      val filteredMovies =  moviePairSimilarities.filter(x=>{
        val pair = x._1
        val sim = x._2

        (pair._1 == movieId || pair._2==movieId) && (sim._1>scoreThreshold && sim._2>coOcurrenceThreshold)

      })

      var results = filteredMovies.map(x=>(x._2, x._1)).sortByKey(false).take(10)

      println("\nTop 10 similar movies for " + movieNamedDict(movieId))
      for(result <- results){
        val sim = result._1
        val pair = result._2

        var similarMovieId = pair._1
        if(pair._1==movieId){
          similarMovieId = pair._2
        }

        println(movieNamedDict(similarMovieId)+"\tscore: "+sim._1+"\tstrength:"+sim._2)


      }
    }

  }

}
