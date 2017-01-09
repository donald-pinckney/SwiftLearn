//
//  NeuralNetwork.swift
//  AES-NN
//
//  Created by Donald Pinckney on 1/1/17.
//
//

import Linear
import Dispatch
import Foundation

public struct NeuralNetwork {
    public var layers: [NeuralNetworkLayer]
    
    public init(layers: [NeuralNetworkLayer]) {
        self.layers = layers
    }
    
    public mutating func forwardPropagate(_ X: Matrix) -> Matrix {
        var currentInput = X
        for l in 0..<layers.count {
            layers[l].forwardPropagate(inputs: currentInput)
            currentInput = layers[l].currentState.activations
        }
        
        return currentInput
    }
    
    // X is (in x m), Y is (out x m),
    // where in = # of input dimensions, out = # of output dimensions, m = # of training examples.
    private mutating func cost(X: Matrix, Y: Matrix, lambda: Double) -> Double {
        let m = Double(X.width)
        let H = forwardPropagate(X)
        
        var regularizationSum = 0.0
        
        for l in layers {
            let params = l.weights[0..<l.weights.height, 1..<l.weights.width]
            if lambda != 0 {
                regularizationSum += (params .* params).sumAll()
            }
        }
        
        return -1/m * (Y .* log(H) + (1-Y) .* log(1-H)).sumAll() + lambda / (2*m) * regularizationSum
    }
    
    // x and y should be column vectors
    // Precondition, the network has already been forwardPropagated, with x in index exampleIndex
    private mutating func backPropagateSingle(x: Matrix, y: Matrix, exampleIndex: Int) -> [Matrix] {
        var derivatives = [Matrix](repeating: Matrix.zeros(1, 1), count: layers.count)

        for l in (0..<layers.count).reversed() {
            if l == layers.count - 1 {
                layers[l].backPropagate(finalCorrectOutputs: y, exampleIndex: exampleIndex)
            } else {
                layers[l].backPropagate(backMultipliedErrorsOfForwardLayer: layers[l+1].currentState.backMultipliedErrors, exampleIndex: exampleIndex)
            }
            derivatives[l] = layers[l].currentState.outputErrors * layers[l].currentState.inputsT[exampleIndex, 0..<layers[l].currentState.inputsT.width]
            derivatives[l] .*= layers[l].constantMask
        }
        
        return derivatives
    }
    
    public mutating func backPropagate(X: Matrix, Y: Matrix, lambda: Double) -> (cost: Double, derivative: [Matrix]) {
        
        let cost = self.cost(X: X, Y: Y, lambda: lambda) // This also performs the forwardPropagate
        let M = X.width
        
        let NUM_THREADS = 8

        let derivativeSumBlank: ContiguousArray<Matrix> = ContiguousArray(layers.map { layer in
            Matrix.zeros(layer.weights.height, layer.weights.width)
        })
        
        var derivativeSums: ContiguousArray<ContiguousArray<Matrix>> = ContiguousArray(repeating: derivativeSumBlank, count: NUM_THREADS)
        
        
//        let xCols = X.columns
//        let yCols = Y.columns
        let layerCount = layers.count
        
        let XT = X.T
        let YT = Y.T

        DispatchQueue.concurrentPerform(iterations: NUM_THREADS) { threadIndex in
            
            var network = self

            let mThread: Int
            if threadIndex < NUM_THREADS - 1 {
                mThread = M / NUM_THREADS
            } else {
                mThread = M - (M / NUM_THREADS * (NUM_THREADS - 1))
            }
            
            let startIndex = threadIndex * (M / NUM_THREADS)
            let endIndex = startIndex + mThread
            
            for i in startIndex..<endIndex {
//                let xc = xCols[i]
//                let yc = yCols[i]
//                let xc = X[0..<X.height, i]
//                let yc = Y[0..<Y.height, i]
                let startXDataIdx = i * XT.width
                let endXDataIdx = (i + 1) * XT.width
                let startYDataIdx = i * YT.width
                let endYDataIdx = (i + 1) * YT.width
                let xc = Matrix(rowMajorData: Array(XT.data[startXDataIdx..<endXDataIdx]), width: 1)
                let yc = Matrix(rowMajorData: Array(YT.data[startYDataIdx..<endYDataIdx]), width: 1)

                let deriv = network.backPropagateSingle(x: xc, y: yc, exampleIndex: i)
                for l in 0..<layerCount {
                    derivativeSums[threadIndex][l] += deriv[l]
                }
            }
            
            
//            var network = self
//            
//            let xc = xCols[threadIndex]
//            let yc = yCols[threadIndex]
//            let deriv = network.backPropagateSingle(x: xc, y: yc, exampleIndex: threadIndex)
//            for l in 0..<layerCount {
//                derivativeSums[threadIndex][l] += deriv[l]
//            }
        }
        
        var derivatives = derivativeSums[0]
        for i in 1..<NUM_THREADS {
            for l in 0..<layerCount {
                derivatives[l] += derivativeSums[i][l]
            }
        }
        
        for l in 0..<layerCount {
            derivatives[l] /= Double(M)
            
            if lambda != 0 {
                let allRows: Range<Int> = 0..<derivatives[l].height
                let noBiasColumn: Range<Int> = 1..<derivatives[l].width
                derivatives[l][allRows, noBiasColumn] += lambda / Double(M) * layers[l].weights[allRows, noBiasColumn]
            }
        }
        
        
//        Dispatch
        
        return (cost, Array(derivatives))
    }
    
    
    public func getAllWeights() -> Matrix {
        return Matrix.unrollToColumnVector(Xs: layers.map { layer in layer.weights }).column
    }
    
    public mutating func setAllWeights(_ weightsColumn: Matrix) {
        let weights = Matrix.rollColumnVectorToMatrices(column: weightsColumn, sizes: layers.map { ($0.weights.height, $0.weights.width) })
        
        for i in 0..<layers.count {
            layers[i].weights = weights[i]
        }
    }
    
//    // Don't use this for actual machine learning, only debugging
//    // TODO: Refactor into separate routine independent of the model.
//    public mutating func numericalDerivative(X: Matrix, Y: Matrix, lambda: Double) -> (cost: Double, derivative: [Matrix]) {
//        let oldWeights = getAllWeights()
//        
//        let cost = self.cost(X: X, Y: Y, lambda: lambda)
//        
//        var derivs = layers.map { l in
//            Matrix.zeros(l.weights.height, l.weights.width)
//        }
//        
//        let e = 1e-4
//        
//        var tweakedWeightsLeft = oldWeights
//        var tweakedWeightsRight = oldWeights
//
//        for l in 0..<layers.count {
//            for r in 0..<layers[l].weights.height {
//                for c in 0..<layers[l].weights.width {
//                    let old = tweakedWeightsLeft[l][r, c]
//
//                    tweakedWeightsLeft[l][r, c] -= e
//                    tweakedWeightsRight[l][r, c] += e
//
//                    setAllWeights(tweakedWeightsLeft)
//                    let costLeft = self.cost(X: X, Y: Y, lambda: lambda)
//                    
//                    setAllWeights(tweakedWeightsRight)
//                    let costRight = self.cost(X: X, Y: Y, lambda: lambda)
//                    
//                    derivs[l][r, c] = (costRight - costLeft) / (2*e)
//                    
//                    tweakedWeightsLeft[l][r, c] = old
//                    tweakedWeightsRight[l][r, c] = old
//                }
//            }
//        }
//        
//        
//        setAllWeights(oldWeights)
//        
//        return (cost, derivs)
//    }
}

