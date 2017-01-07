//
//  NeuralNetworkOptimizable.swift
//  AES-NN
//
//  Created by Donald Pinckney on 1/4/17.
//
//

import Linear
import Optimization

public struct NeuralNetworkOptimizable: Optimizable {
    
    public var network: NeuralNetwork
    let X: Matrix
    let Y: Matrix
    public let batchSize: Int
    public let lambda: Double
    var currentTrainingIndex: Int = 0
        
    public init(network: NeuralNetwork, X: Matrix, Y: Matrix, lambda: Double = 0, batchSize: Int = -1) {
        precondition(X.width == Y.width)
        
        if batchSize == -1 {
            self.batchSize = X.width
        } else {
            self.batchSize = batchSize
        }
        
        self.network = network
        self.X = X
        self.Y = Y
        self.lambda = lambda
    }
    
    public func initialParameters() -> Matrix {
        return network.getAllWeights()
    }

    mutating public func costFunction(_ P: Matrix) -> (cost: Double, derivative: Matrix) {
//        let numDer = network.numericalDerivative(X: X, Y: Y)
        
        network.setAllWeights(P)
        
        let endIndex = min(currentTrainingIndex + batchSize, X.width)
        
        let Xs = X[0..<X.height, currentTrainingIndex..<endIndex]
        let Ys = Y[0..<Y.height, currentTrainingIndex..<endIndex]
        
        currentTrainingIndex += batchSize
        if currentTrainingIndex >= X.width {
            currentTrainingIndex = 0
        }
        
        let (cost, derivs) = network.backPropagate(X: Xs, Y: Ys, lambda: lambda)
        return (cost, Matrix.unrollToColumnVector(Xs: derivs).column)
//        return numDer
    }


    
}
