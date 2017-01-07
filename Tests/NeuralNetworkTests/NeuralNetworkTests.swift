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
    
    
    func testMNISTLearn() {
        let hiddenLayer = NeuralNetworkLayer(inputNodeCount: 400, outputNodeCount: 25, activationFunction: .sigmoid)
        let outputLayer = NeuralNetworkLayer(inputNodeCount: 25, outputNodeCount: 10, activationFunction: .sigmoid)
        
        var NN = NeuralNetwork(layers: [hiddenLayer, outputLayer])
        
        let testBundle = Bundle(for: type(of: self))
        let X_URL = testBundle.url(forResource: "MNIST_X", withExtension: "csv")!
        let Y_URL = testBundle.url(forResource: "MNIST_Y", withExtension: "csv")!

        let X = Matrix(csvURL: X_URL)
        let Y = Matrix(csvURL: Y_URL)

        
        let networkOptimizable = NeuralNetworkOptimizable(network: NN, X: X, Y: Y, lambda: 1)
        let gradDescent = GradientDescentOptimizer(learningRate: 3, precision: -1, maxIterations: 1000)
        let (history, optimum) = gradDescent.optimize(networkOptimizable)
        
        NN.setAllWeights(optimum)
        
        let learnedXOR = NN.forwardPropagate(X)
        XCTAssertEqualWithAccuracy(learnedXOR, Y, accuracy: 0.01)
        
        XCTAssertEqualWithAccuracy(history.last!, 0, accuracy: 0.01)
        
    }


    static var allTests : [(String, (NeuralNetworkTests) -> () throws -> Void)] {
        return [
            ("testXORLearn", testXORLearn),
            ("testMNISTLearn", testMNISTLearn)
        ]
    }
}
