import org.apache.spark.broadcast.Broadcast
import org.apache.spark.Accumulator
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vectors
import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.rdd.RDD
import scala.util.Random
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.AccumulatorParam
import org.apache.spark.mllib.clustering.KMeans

//custom breeze vector accumulator
object VectorAccumulatorParam extends AccumulatorParam[BDV[Double]] {
	def zero(v: BDV[Double]): BDV[Double] = v
	def addInPlace(v1: BDV[Double], v2: BDV[Double]): BDV[Double] = v1 += v2
}


//initilize particle swarm
def initParticles(data:RDD[BDV[Double]], numClusters:Int, numParticles:Int): Array[(Array[BDV[Double]],Array[BDV[Double]], Array[BDV[Double]], Double)] = { 
	//Temp particle storage while they are created
	val particleBuild = new ArrayBuffer[(Array[BDV[Double]],Array[BDV[Double]], Array[BDV[Double]], Double)]();
	val dimensions = data.first.size
    var step = 0
	while (step < numParticles){
		//create a new particle with cluster located at random data points in dataset
		val clusters = data.takeSample(false, numClusters, Random.nextLong())
		val vel = Array.fill(numClusters)(new BDV(Array.fill(dimensions)(0.0)))
		val pb = clusters.clone
		//particle object (note error set to zero (max error))
		val particle = (clusters, vel, pb, Double.PositiveInfinity)
		//add particle to array with all particles
		particleBuild += particle;

		step = step +1;
	}
	return particleBuild.toArray
}

def runAlgorithm(iter:Int, data:RDD[BDV[Double]], swarm:Array[(Array[BDV[Double]],Array[BDV[Double]], Array[BDV[Double]], Double)], numClusters:Int, numParticles:Int): (Array[BDV[Double]], Double) = {
  
  val const1 = 1.49
  val const2 = 1.49
  val inertia = 0.4
	val particles: Array[(Array[BDV[Double]],Array[BDV[Double]], Array[BDV[Double]], Double)] = swarm
	
	var iterations = 0
	var overallError = 0.0
  val sparkVectData = data.map(x=> Vectors.dense(x.toArray))
	val kmclusters = KMeans.train(sparkVectData, numClusters, 3)
  var minSwarmError = kmclusters.computeCost(sparkVectData)
  var gBest = kmclusters.clusterCenters.map(x=> BDV(x.toArray))

	var bc = sc.broadcast(particles)

	while (iterations < iter){
		//particles
		val p = bc.value
		//errors for each particle
		val errors =  sc.accumulator(BDV.zeros[Double](numParticles))(VectorAccumulatorParam)

		data.mapPartitions( points =>
           points.map{ d2 =>
              errors.add(BDV(p.map{ case (pos, vel, pb, err) => 
                var bestDistance:Double = Double.PositiveInfinity
                pos.foreach{ d1 => 
                  val dist = (d1 - d2) dot (d1 - d2)
                  if(dist<bestDistance){
                    bestDistance = dist
                  }
                }
                bestDistance
             }))
        }).collect()

        val sums = errors.value.toArray

        bc.destroy

        //returns a data structure with each particles new best position, new best error, and index
      val pbs = particles.zipWithIndex.map{ case((pos, vel, pb, err),i) =>
          var nErr = err
          var npb = pb
          if(sums(i)<err){
              if( sums(i) < minSwarmError ){
                minSwarmError = sums(i)
                gBest = pos
              }
              nErr = sums(i)
              npb = pos
          }
          val upP:Array[(BDV[Double], BDV[Double])] = pos.zipWithIndex.map{ case(x:BDV[Double], k:Int) =>
              val newVel = (npb(k) - x)*Random.nextDouble()*const1 + (gBest(k) - x)*Random.nextDouble()*const2 + vel(k)*inertia
              val newPos = x + newVel
              (newPos, newVel)
          }
          val newP = upP.map{ case(x,y) => x}
          val newV = upP.map{ case(x,y) => y}
         //return a new particle with new position and velocity vects
         (newP, newV, npb, nErr)
      }

      bc = sc.broadcast(pbs)
      overallError = minSwarmError
      iterations = iterations + 1
	}
	return (gBest, overallError)

}

def PSO (data:RDD[BDV[Double]], numClusters:Int, numParticles:Int, iterations:Int): (Array[BDV[Double]], Double) = {
  val swarm = initParticles(data, numClusters, numParticles)
  val gBest = runAlgorithm(iterations, data, swarm, numClusters, numParticles)
    return gBest
}