import org.apache.spark.broadcast.Broadcast
import org.apache.spark.Accumulator
import breeze.linalg._
import scala.util.Random
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.AccumulatorParam


//custom breeze vector accumulator
object VectorAccumulatorParam extends AccumulatorParam[DenseVector[Double]] {
	def zero(v: DenseVector[Double]): DenseVector[Double] = v
	def addInPlace(v1: DenseVector[Double], v2: DenseVector[Double]): DenseVector[Double] = v1 += v2
}

def PSO(data:RDD[DenseVector[Double]], numClusters:Int, numParticles:Int, acceptableError:Double): (Array[DenseVector[Double]], Double) = {
	val swarm = initParticles(data, numClusters, numParticles)
	val gBest = runAlgorithm(acceptableError, data, swarm, numClusters, numParticles)
    return gBest
}
//initilize particle swarm
def initParticles(data:RDD[DenseVector[Double]], numClusters:Int, numParticles:Int): Array[(Array[DenseVector[Double]],Array[DenseVector[Double]], Array[DenseVector[Double]], Double)] = { 
	//Temp particle storage while they are created
	val particleBuild = new ArrayBuffer[(Array[DenseVector[Double]],Array[DenseVector[Double]], Array[DenseVector[Double]], Double)]();
	val dimensions = data.first.size
    var step = 0
	while (step < numParticles){
		//create a new particle with cluster located at random data points in dataset
		val clusters = data.takeSample(false, numClusters, Random.nextLong())
		val vel = Array.fill(numClusters)(new DenseVector(Array.fill(dimensions)(0.0)))
		val pb = clusters.clone
		//particle object (note error set to zero (max error))
		val particle = (clusters, vel, pb, Double.PositiveInfinity)
		//add particle to array with all particles
		particleBuild += particle;

		step = step +1;
	}
	return particleBuild.toArray
}

def runAlgorithm(acceptableError:Double, data:RDD[DenseVector[Double]], swarm:Array[(Array[DenseVector[Double]],Array[DenseVector[Double]], Array[DenseVector[Double]], Double)], numClusters:Int, numParticles:Int): (Array[DenseVector[Double]], Double) = {
  
    val const1 = 2.25
    val const2 = 2.35
    val inertia = 0.4
	val particles: Array[(Array[DenseVector[Double]],Array[DenseVector[Double]], Array[DenseVector[Double]], Double)] = swarm
	var minSwarmError:Double = Double.PositiveInfinity
	var iterations = 0
	var overallError = 0.0
	var gBest = data.takeSample(false, numClusters, Random.nextLong());
	var bc = sc.broadcast(particles)

	while (iterations < 10){
		//particles
		val p = bc.value
		//errors for each particle
		val errors =  sc.accumulator(DenseVector.zeros[Double](numParticles))(VectorAccumulatorParam)

		data.mapPartitions( points =>
           points.map{ d2 =>
              errors.add(DenseVector(p.map{ case (pos, vel, pb, err) => 
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
              if(sums(i)<minSwarmError){
                minSwarmError = sums(i)
                gBest = pos
              }
              nErr = sums(i)
              npb = pos
          }
          val upP:Array[(DenseVector[Double], DenseVector[Double])] = pos.zipWithIndex.map{ case(x:DenseVector[Double], k:Int) =>
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

def totalCost(data:RDD[DenseVector[Double]], gbest:Array[DenseVector[Double]]): Double = {
  val cost = sc.accumulator(1.1);
  data.mapPartitions(points =>
    points.map{ d =>
      var bestDistance: Double = Double.PositiveInfinity 
      gbest.foreach{cluster => 
        var dist = 0.0
        val arr = cluster.toArray
        val arr2 = d.toArray
        for(i <- 0 to arr.length-1){
          var score = arr(i) - arr2(i)
          dist += score*score
          print(dist)
        }
        if( dist < bestDistance){
          bestDistance = dist
        }
      }
      cost.add(bestDistance);
    }
  )
  return math.sqrt(cost.value)
}