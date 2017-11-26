/***
Matthew Sherar
CISC500
Queen's Computing
***/

import org.apache.spark.Accumulator
import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.AccumulatorParam

//custom breeze vector accumulator
object VectorAccumulatorParam extends AccumulatorParam[BDV[Double]] {
  def zero(v: BDV[Double]): BDV[Double] = v
  def addInPlace(v1: BDV[Double], v2: BDV[Double]): BDV[Double] = v1 += v2
}
import scala.util.Random
import org.apache.spark.rdd.RDD
//Main Particle class
class Particle(numC: Int, d: Array[Array[Double]], numD:Int, numData: Long) extends java.io.Serializable {
	import org.apache.spark.Accumulator
	import breeze.linalg.{DenseVector => BDV}
	import org.apache.spark.AccumulatorParam
	import org.apache.spark.rdd.RDD
	import scala.util.Random
	var data = sc.parallelize(d)
	val numDimensions = numD
	var Pc = 0.0
	val numClusters = numC
	//Initializing particle
	var weightMat = Array.fill(numClusters, numDimensions) { Random.nextDouble() }
	var pbestWeightMat = weightMat.clone
	var vel = Array.fill(numClusters, numDimensions) { Random.nextDouble()*0.25 }//one particle velocity matrix
	var clusterCentroids = data.takeSample(false, numClusters, Random.nextLong())
	var bestCentroids = clusterCentroids;
	var dM = Array.ofDim[Int](numData.toInt, numClusters); //data memberships
	var bestdM = dM
	var fitnessVal = -1.0;
	var bestFitness = -1.0;
	var beta = 8.0
	
	val c1 = 1.49
	val c2 = 1.49

	def centroidUpdate(data:RDD[Array[Double]])  {
		val centroids = Array.ofDim[Double](numClusters, numDimensions);
		val mb = sc.broadcast(dM);
		
		for( k <- 0 to numClusters-1){
			val allpoints =  sc.accumulator(BDV.zeros[Double](numDimensions))(VectorAccumulatorParam)
			val numpoints = sc.accumulator(0)
			data.zipWithIndex.map{ case(x, i:Long) =>
				val mem = mb.value
				if(mem(i.toInt)(k) == 1){
					allpoints.add(BDV(x.toArray))
					numpoints.add(1)
				}
			}.collect
			if(numpoints.value == 0){
				centroids(k) = Array.fill(numDimensions)(0.0)
			}
			else{
				centroids(k) = (allpoints.value/numpoints.value.toDouble).toArray
			}
		}
		clusterCentroids = centroids;
	}

	def bestUpdate(){
		if(fitnessVal > bestFitness){
			bestFitness = fitnessVal;
			pbestWeightMat = weightMat
			bestCentroids = clusterCentroids
			bestdM = dM

		}
	}

}

object Particle{def apply(numC: Int, d: Array[Array[Double]], numD:Int, numData: Long) = new Particle(numC: Int, d: Array[Array[Double]], numD:Int, numData: Long)}

def psovw(numP: Int, d:Array[Array[Double]], numD:Int, numC:Int, iterations: Int): (Array[Array[Double]], Double, Particle) =  {

	val data = sc.parallelize(d)
	val swarm: Array[Particle] = Array.ofDim[Particle](numP);
	val tData = data.count;
	var inertia = 0.0
	var iMin = 0.05
	val beta = 4
	var iMax = 0.9

	for(i <- 0 to numP-1){
		val p = new Particle(numC, data.collect, numD, tData)
		p.Pc = 0.05 + 0.45*((math.exp((10*(i-1))/numP)-1)/(math.exp(10)-1))
		swarm(i) = p

	}
	var pFitness: Array[Double] = Array.ofDim[Double](numP);
	var maxFitness = -1.0;
	var gIndex = 0 ;
	var gBest = swarm(0).clusterCentroids;

	
	for(i <- 0 to iterations-1){
		inertia = ((iMax + (-1)*iMin)/2.0)*math.cos(3.0*math.Pi*(i.toDouble/iterations.toDouble)) + (iMin + iMax)/2.0
		//update particle memberships
		var sbc = sc.broadcast(swarm);
		val dataMemberships: Array[Array[Array[Int]]] = data.map{ x =>
			val sbcV = sbc.value;
			val dmem: Array[Array[Int]] = Array.ofDim[Int](numP, numC);
			for(m <- 0 to numP-1){
				var cost = -1.0
				var member = 0
				val xarr = x.toArray 
				for(k <- 0 to numC-1){ 
					var dist = 0.0
					val weightSum = sbcV(m).weightMat(k).sum 
					val wMat = sbcV(m).weightMat(k)
					var xdotz = 0.0 
					var xnorm = 0.0
					var znorm = 0.0
					var w = 0.0
					for (n <- 0 to numD-1){
						w = Math.pow((sbcV(m).weightMat(k)(n)/weightSum), beta)
						xdotz = xdotz + w*w*xarr(n)*sbcV(m).clusterCentroids(k)(n)
						xnorm = xnorm + Math.pow(xarr(n), 2)*w
						znorm  = Math.pow(sbcV(m).clusterCentroids(k)(n), 2)*w
					} 
					dist = (xdotz)/(Math.pow(xnorm, 0.5) + Math.pow(znorm, 0.5) - xdotz)
					if( dist > cost){
					 cost = dist 
					 member = k 
					 }
				}
				var uMat:Array[Int] = Array.fill(numC)(0);
				uMat(member) = 1;
				dmem(m) = uMat;
			}
			dmem
		}.collect
		sbc.destroy
		for(m <- 0 to numP-1){
			for(n <- 0 to tData.toInt-1){
				swarm(m).dM(n) = dataMemberships(n)(m)
			}
		}
		//update centroids
		for(j <- 0 to numP-1){
			swarm(j).centroidUpdate(data);
		}

		//updates all particles fitness
		sbc = sc.broadcast(swarm)
		val errors =  sc.accumulator(BDV.zeros[Double](numP))(VectorAccumulatorParam)
		data.zipWithIndex.map{case (x:Array[Double], w: Long) =>
			val sbcV = sbc.value 
			val errArr = Array.ofDim[Double](numP)
			for(m <- 0 to numP-1){
				var cost = 0.0 
				val xarr = x.toArray 
				for(k <- 0 to numC-1){ 
					val weightSum = sbcV(m).weightMat(k).sum
					if (sbcV(m).dM(w.toInt)(k) == 1){
						val wMat = sbcV(m).weightMat(k)
						var xdotz = 0.0 
						var xnorm = 0.0
						var znorm = 0.0
						var w = 0.0
						for (n <- 0 to numD-1){
							w = Math.pow((sbcV(m).weightMat(k)(n)/weightSum), beta)
							xdotz = xdotz + w*w*xarr(n)*sbcV(m).clusterCentroids(k)(n)
							xnorm = xnorm + Math.pow(xarr(n), 2)*w
							znorm  = Math.pow(sbcV(m).clusterCentroids(k)(n), 2)*w
						} 
						var dist = (xdotz)/(Math.pow(xnorm, 0.5) + Math.pow(znorm, 0.5) - xdotz)
						cost = cost + dist
						
					}
				}
				errArr(m) = cost
			}
			errors.add(BDV(errArr))
		}.collect
		sbc.destroy
		val pFitness = errors.value.toArray;
		for(m <- 0 to numP-1){
			swarm(m).fitnessVal = pFitness(m) 
		}
		
		for(j <- 0 to numP-1){
			//swarm(j).centroidUpdate(data);
			//pFitness(j) = swarm(j).fitness(data);
			if(pFitness(j) > maxFitness){
				gBest = swarm(j).clusterCentroids;
				maxFitness = pFitness(j)
				gIndex = j
			}
			swarm(j).bestUpdate();
			//Particle swarm weight updates using comprehensive best learning
			for(z <- 0 to numC-1){
				for(m <- 0 to numD-1){
					val rp1 = Random.nextDouble();
					if(rp1 > swarm(j).Pc){
						swarm(j).vel(z)(m) = 0.7*swarm(j).vel(z)(m) + 1.49*(swarm(j).pbestWeightMat(z)(m) - swarm(j).weightMat(z)(m)) 
						swarm(j).weightMat(z)(m) = swarm(j).vel(z)(m) + swarm(j).weightMat(z)(m)
					}
					else{
						val rand1 = Random.nextInt(numP-1);
						val rand2 = Random.nextInt(numP-1);
						val choosen = if(swarm(rand1).fitnessVal > swarm(rand2).fitnessVal) rand1 else rand2
						swarm(j).vel(z)(m) = 0.7*swarm(j).vel(z)(m) + 1.49*(swarm(choosen).pbestWeightMat(z)(m) - swarm(j).weightMat(z)(m)) 
						swarm(j).weightMat(z)(m) = swarm(j).vel(z)(m) + swarm(j).weightMat(z)(m)
						
					}
					if(swarm(j).weightMat(z)(m) > 1){
						swarm(j).weightMat(z)(m) = 1;
					}
					if(swarm(j).weightMat(z)(m)<0){
						swarm(j).weightMat(z)(m) = 0
					}
				}	
			}
			
			
		}
		println(maxFitness)
	}
	return (gBest, maxFitness, swarm(gIndex))
	//return (gBest, minFitness)

}