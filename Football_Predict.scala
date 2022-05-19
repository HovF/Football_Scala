val FootballFile =  sc.textFile("file:///home/user409/bd-sp-2017/results.csv")
val IndexFile = sc.textFile("file:///home/user409/bd-sp-2017/Country_index_3.csv")

def isHeader(line: String): Boolean = {line.contains("tournament")}

case class FootballRecord(home_team: String, away_team: String, country: String, results: Float)

def parse(line: String) = {
     val pieces = line.split(',')
     val home_score = pieces(3).toFloat
     val away_score = pieces(4).toFloat
     val results = if(home_score > away_score) 1 else if(away_score> home_score) 0 else 2
     val home_team = pieces(1).toString
     val away_team = pieces(2).toString
     val country = pieces(7).toString
      FootballRecord(home_team, away_team, country, results)
      }

case class IndexParse(home_team: String, index: Float)

def parse2(line: String) = {
     val pieces = line.split(',')
     val home_team = pieces(0).toString
     val index = pieces(1).toFloat
     IndexParse(home_team, index)
      }

val Footpars = FootballFile.filter(x => !isHeader(x)).map(parse)

val FootData = Footpars.filter(x => !(x.results == 2))

val IndexData = IndexFile.map(parse2)

// Index Home Team, Away Team and Country to host the game 

val mappedFoot = FootData.map(item => (item.home_team, item))

val mappedIndex = IndexData.map(item => (item.home_team, item))

val j_1 = mappedIndex.join(mappedFoot)

val m_1 = j_1.map(x => (x._2._2.away_team, x._2._2.country ,  x._2._2.results, x._2._1.index))

val m_11 = m_1.map(item => (item._1, item))

val j_2 = mappedIndex.join(m_11)

// Country, results, index_home, index_away

val m_2 = j_2.map(x => (x._2._2._2 ,x._2._2._3,x._2._2._4, x._2._1.index))

val m_22 = m_2.map(item => (item._1, item))

val j_3 = mappedIndex.join(m_22)

//m_3 RDD [results, home index, away index, country_index]

val m_3 =j_3.map(x => (x._2._2._2 ,x._2._2._3,x._2._2._4, x._2._1.index))

val splits = m_3.randomSplit(Array(0.7, 0.3), seed = 11L)

val Training_Data = splits(0).cache()
val Test_Data = splits(1)

// Start Analysis

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.{StandardScaler, StandardScalerModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}


val parsedData = Training_Data.map(x => LabeledPoint(x._1.toDouble, Vectors.dense(x._2.toDouble, x._3.toDouble, x._4.toDouble)))

val parsed_Test_Data = Test_Data.map(x => LabeledPoint(x._1.toDouble, Vectors.dense(x._2.toDouble, x._3.toDouble, x._4.toDouble)))


val scaler1 = new StandardScaler().fit(parsedData.map(x => x.features))

val Scaled_data = parsedData.map(x => LabeledPoint(x.label, scaler1.transform(x.features)))
val Scaled_Test_data = parsed_Test_Data.map(x => LabeledPoint(x.label, scaler1.transform(x.features)))


val numIterations = 100

val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(Scaled_data)

val model_2 = SVMWithSGD.train(Scaled_data, numIterations)


val predictionAndLabels = Scaled_Test_data.map {x =>
  val prediction = model.predict(x.features)
  (prediction, x.label)
}


val predictionAndLabels_2 = Scaled_Test_data.map {x =>
  val prediction_2 = model_2.predict(x.features)
  (prediction_2, x.label)
}


// Get evaluation metrics.
val metrics = new MulticlassMetrics(predictionAndLabels)
val accuracy = metrics.accuracy
println(s"Accuracy = $accuracy")

// Get evaluation metrics.
val metrics_2 = new MulticlassMetrics(predictionAndLabels_2)
val accuracy_2 = metrics_2.accuracy
println(s"Accuracy = $accuracy_2")

