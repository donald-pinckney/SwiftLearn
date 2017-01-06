import XCTest
@testable import NeuralNetwork
import Linear
import Optimization
import Plotting

class NeuralNetworkTests: XCTestCase {
    func testXORLearn() {
        let hiddenLayer = NeuralNetworkLayer(inputNodeCount: 2, outputNodeCount: 2, activationFunction: .sigmoid)
        let outputLayer = NeuralNetworkLayer(inputNodeCount: 2, outputNodeCount: 2, activationFunction: .sigmoid)
        
        var xorNN = NeuralNetwork(layers: [hiddenLayer, outputLayer])
        
        let X = Matrix(format: [
            [0, 0, 1, 1],
            [0, 1, 0, 1]
            ])
        
        let Y = Matrix(format: [
            [0, 1, 1, 0],
            [1, 0, 0, 1]
            ])
        
        let networkOptimizable = NeuralNetworkOptimizable(network: xorNN, X: X, Y: Y)
        let gradDescent = GradientDescentOptimizer(learningRate: 3, precision: -1, maxIterations: 2000)
        let (history, optimum) = gradDescent.optimize(networkOptimizable)
        
        xorNN.setAllWeights(optimum)
        
        let learnedXOR = xorNN.forwardPropagate(X)
        XCTAssertEqualWithAccuracy(learnedXOR, Y, accuracy: 0.01)
        
        XCTAssertEqualWithAccuracy(history.last!, 0, accuracy: 0.01)
    }


    static var allTests : [(String, (NeuralNetworkTests) -> () throws -> Void)] {
        return [
            ("testXORLearn", testXORLearn),
        ]
    }
}
