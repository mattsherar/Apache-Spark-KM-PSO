import org.apache.spark.broadcast.Broadcast
import org.apache.spark.Accumulator
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vectors
import breeze.linalg.{DenseVector => BDV}
import breeze.linalg.functions.euclideanDistance
import breeze.linalg._
import org.apache.spark.rdd.RDD
import scala.util.Random
 import math._
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.AccumulatorParam
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.clustering.BisectingKMeans

//custom breeze vector accumulator
object VectorAccumulatorParam extends AccumulatorParam[BDV[Double]] {
    def zero(v: BDV[Double]): BDV[Double] = v
    def addInPlace(v1: BDV[Double], v2: BDV[Double]): BDV[Double] = v1 += v2
}


//initilize particle swarm
def initParticles(data:RDD[BDV[Double]], numClusters:Int, numParticles:Int): Array[(Array[BDV[Double]], Array[Double] ,Array[BDV[Double]], Array[BDV[Double]], Double)] = { 
    
    //Temp particle storage while they are created
    val particleBuild = new ArrayBuffer[(Array[BDV[Double]], Array[Double],Array[BDV[Double]], Array[BDV[Double]], Double)]();
    val dimensions = data.first.size
    var step = 0

    while (step < numParticles){
        //create a new particle with cluster located at random data points in dataset
        val clusters = data.takeSample(false, numClusters, Random.nextLong())
        val cnorm = clusters.map(x=>norm(x))
        val vel = Array.fill(numClusters)(new BDV(Array.fill(dimensions)(0.0)))
        val pb = clusters.clone
        //particle object (note error set to zero (max error))
        val particle = (clusters, cnorm, vel, pb, Double.PositiveInfinity)
        //add particle to array with all particles
        particleBuild += particle;

        step = step +1;
    }

    //Initialize K-Means clusters to Global Best Particle
   //
    //particleBuild += p1
    return particleBuild.toArray
}

def runAlgorithm(iter:Int, data:RDD[BDV[Double]], swarm:Array[(Array[BDV[Double]],Array[Double],Array[BDV[Double]], Array[BDV[Double]], Double)], numClusters:Int, numParticles:Int): (Array[BDV[Double]], Double, Array[Double]) = {
  
  //Constants used in PSO algorithm
  val const1 = 1.49
  val const2 = 1.49
  var inertia = 0.4
  val C = 3.0
  val iMax = 0.9
  val iMin = 0.05


  val particles: Array[(Array[BDV[Double]], Array[Double],Array[BDV[Double]], Array[BDV[Double]], Double)] = swarm
  val itResults = Array.fill(iter)(0.0)
  var iterations = 0
  var overallError = 0.0


  val sparkVectData = data.map(x=> Vectors.dense(x.toArray))
  val kmclusters = KMeans.train(sparkVectData, numClusters, 60)
  var gBest = kmclusters.clusterCenters.map(x=>new BDV(x.toArray))
  var minSwarmError = data.mapPartitions( points =>
    points.map{ d2 =>
      var bestDistance:Double = Double.PositiveInfinity
      gBest.map{ d1 => 
        val dist = euclideanDistance(d1, d2)
        if(dist < bestDistance){
          bestDistance = dist
        }
      }
      bestDistance
  }).sum()

  var bc = sc.broadcast( particles )
  val norms = data.map( x => norm(x) )
  val zippD = data.zip(norms)

  while (iterations < iter){
      //particles
      val p = bc.value
      inertia = ((iMax + (-1)*iMin)/2.0)*math.cos(1.0*math.Pi*(iterations.toDouble/iter.toDouble)) + (iMin + iMax)/2.0
      //errors for each particle
      val errors =  sc.accumulator(BDV.zeros[Double](numParticles))(VectorAccumulatorParam)

      zippD.mapPartitions( points =>
         points.map{ case (d2, n2) =>
            errors.add(BDV(p.map{ case (pos, norm, vel, pb, err) => 
              var bestDistance:Double = Double.PositiveInfinity
              pos.zipWithIndex.foreach{ case(d1,i) => 
                val dist = euclideanDistance(d1, d2)
                if(dist < bestDistance){
                  bestDistance = dist
                }
              }
              bestDistance
           }))
      }).collect()

      val sums = errors.value.toArray


      bc.destroy
     // val pbs= new ArrayBuffer[(Array[BDV[Double]],Array[BDV[Double]], Array[BDV[Double]], Double)]();
      //returns a data structure with each particles new best position, new best error, and index
  
    val pbs = particles.zipWithIndex.map{ case((pos,norm, vel, pb, err),i) =>
        var nErr = err
        var npb = pb
        if(sums(i) < err){
            if( sums(i) < minSwarmError ){
              minSwarmError = sums(i)
              gBest = pos
            }
            nErr = sums(i)
            npb = pos
        }
        val upP:Array[(BDV[Double], BDV[Double])] = pos.zipWithIndex.map{ case(x:BDV[Double], k:Int) =>
            //val rp1 = Random.nextDouble();
            val newVel = (npb(k) - x)*Random.nextDouble()*const1 + (gBest(k) - x)*Random.nextDouble()*const2 + vel(k)*inertia

            val newPos = x + newVel
            (newPos, newVel)
        }
        val newP = upP.map{ case(x,y) => x}
        val newNorm = newP.map(x =>  breeze.linalg.norm(x))
        val newV = upP.map{ case(x,y) => y}
       //return a new particle with new position and velocity vects
       (newP, newNorm, newV, npb, nErr)
    }
    for(i<- 0 to numParticles-1){
      particles(i)  = pbs(i)
    }
    bc = sc.broadcast(pbs)
    overallError = minSwarmError

    itResults(iterations) = minSwarmError
   
    iterations = iterations + 1
    
  }
    
    return (gBest, overallError, itResults)

}

def PSO (data:RDD[BDV[Double]], numClusters:Int, numParticles:Int, iterations:Int): (Array[BDV[Double]], Double, Array[Double]) = {
  val swarm = initParticles(data, numClusters, numParticles)
  val gBest = runAlgorithm(iterations, data, swarm, numClusters, numParticles)
    return gBest
}