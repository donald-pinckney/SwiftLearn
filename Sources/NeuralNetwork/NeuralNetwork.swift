//
//  NeuralNetwork.swift
//  AES-NN
//
//  Created by Donald Pinckney on 1/1/17.
//
//

import Linear

public struct NeuralNetwork {
    public var layers: [NeuralNetworkLayer]
    
    public init(layers: [NeuralNetworkLayer]) {
        self.layers = layers
    }
    
    public func forwardPropagate(_ X: Matrix) -> Matrix {
        var currentInput = X
        for layer in layers {
            layer.forwardPropagate(inputs: currentInput)
            currentInput = layer.currentState.activations
        }
        
        return currentInput
    }
    
    // X is (in x m), Y is (out x m),
    // where in = # of input dimensions, out = # of output dimensions, m = # of training examples.
    private func cost(X: Matrix, Y: Matrix, lambda: Double) -> Double {
        let m = Double(X.width)
        let H = forwardPropagate(X)
        
        var regularizationSum = 0.0
        
        for l in layers {
            let params = l.weights[0..<l.weights.height, 1..<l.weights.width]
            if lambda != 0 {
                regularizationSum += (params .* params).sumColumns().sumRows()[0]
            }
        }
        
        return -1/m * (Y .* log(H) + (1-Y) .* log(1-H)).sumColumns().sumRows()[0] + lambda / (2*m) * regularizationSum
    }
    
    // x and y should be column vectors
    // Precondition, the network has already been forwardPropagated, with x in index exampleIndex
    private func backPropagateSingle(x: Matrix, y: Matrix, exampleIndex: Int) -> [Matrix] {
        let last = layers.last!
        var derivatives = [Matrix](repeating: Matrix.zeros(1, 1), count: layers.count)

        last.backPropagate(finalCorrectOutputs: y, exampleIndex: exampleIndex)
        derivatives[layers.count - 1] = last.currentState.outputErrors * last.currentState.inputs[0..<last.currentState.inputs.height, exampleIndex].T
        derivatives[layers.count - 1] .*= last.constantMask
        
        for l in (0..<(layers.count - 1)).reversed() {
            layers[l].backPropagate(backMultipliedErrorsOfForwardLayer: layers[l+1].currentState.backMultipliedErrors, exampleIndex: exampleIndex)
            derivatives[l] = layers[l].currentState.outputErrors * layers[l].currentState.inputs[0..<layers[l].currentState.inputs.height, exampleIndex].T
            derivatives[l] .*= layers[l].constantMask
        }
        
        return derivatives
    }
    
    public func backPropagate(X: Matrix, Y: Matrix, lambda: Double) -> (cost: Double, derivative: [Matrix]) {
        let cost = self.cost(X: X, Y: Y, lambda: lambda) // This also performs the forwardPropagate
        let m = X.width

        var derivativeSum = layers.map { layer in
            Matrix.zeros(layer.weights.height, layer.weights.width)
        }
        
        let xCols = X.columns
        let yCols = Y.columns
        for i in 0..<m {
            let xc = xCols[i]
            let yc = yCols[i]
            let deriv = backPropagateSingle(x: xc, y: yc, exampleIndex: i)
            for i in 0..<layers.count {
                derivativeSum[i] += deriv[i]
            }
        }
        
        for i in 0..<layers.count {
            derivativeSum[i] /= Double(m)
            
            if lambda != 0 {
                let allRows: Range<Int> = 0..<derivativeSum[i].height
                let noBiasColumn: Range<Int> = 1..<derivativeSum[i].width
                derivativeSum[i][allRows, noBiasColumn] += lambda / Double(m) * layers[i].weights[allRows, noBiasColumn]
            }
        }
        
        
        return (cost, derivativeSum)
    }
    
    
    public func getAllWeights() -> [Matrix] {
        return layers.map { layer in layer.weights }
    }
    
    public mutating func setAllWeights(_ weights: [Matrix]) {
        for i in 0..<layers.count {
            layers[i].weights = weights[i]
        }
    }
    
    // Don't use this for actual machine learning, only debugging
    // TODO: Refactor into separate routine independent of the model.
    public mutating func numericalDerivative(X: Matrix, Y: Matrix, lambda: Double) -> (cost: Double, derivative: [Matrix]) {
        let oldWeights = getAllWeights()
        
        let cost = self.cost(X: X, Y: Y, lambda: lambda)
        
        var derivs = layers.map { l in
            Matrix.zeros(l.weights.height, l.weights.width)
        }
        
        let e = 1e-4
        
        var tweakedWeightsLeft = oldWeights
        var tweakedWeightsRight = oldWeights

        for l in 0..<layers.count {
            for r in 0..<layers[l].weights.height {
                for c in 0..<layers[l].weights.width {
                    let old = tweakedWeightsLeft[l][r, c]

                    tweakedWeightsLeft[l][r, c] -= e
                    tweakedWeightsRight[l][r, c] += e

                    setAllWeights(tweakedWeightsLeft)
                    let costLeft = self.cost(X: X, Y: Y, lambda: lambda)
                    
                    setAllWeights(tweakedWeightsRight)
                    let costRight = self.cost(X: X, Y: Y, lambda: lambda)
                    
                    derivs[l][r, c] = (costRight - costLeft) / (2*e)
                    
                    tweakedWeightsLeft[l][r, c] = old
                    tweakedWeightsRight[l][r, c] = old
                }
            }
        }
        
        
        setAllWeights(oldWeights)
        
        return (cost, derivs)
    }
}

