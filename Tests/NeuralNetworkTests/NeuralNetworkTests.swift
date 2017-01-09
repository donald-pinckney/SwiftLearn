import XCTest
@testable import NeuralNetwork
import Linear
import Optimization
import Plotting
import GeneralMath

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
        let cg = ConjugateGradientOptimizer(maxIterations: 2000)
        let (history, optimum) = cg.optimize(networkOptimizable)
        
        xorNN.setAllWeights(optimum)
        
        let learnedXOR = xorNN.forwardPropagate(X)
        XCTAssertEqualWithAccuracy(learnedXOR, Y, accuracy: 0.01)
        
        XCTAssertEqualWithAccuracy(history.last!, 0, accuracy: 0.01)
    }
    
    
    func testMNIST() {
        let testBundle = Bundle(for: type(of: self))
        let X_URL = testBundle.url(forResource: "MNIST_X", withExtension: "csv")!
        let Y_URL = testBundle.url(forResource: "MNIST_Y", withExtension: "csv")!
        let hidden_URL = testBundle.url(forResource: "MNIST_layer0_init", withExtension: "csv")!
        let output_URL = testBundle.url(forResource: "MNIST_layer1_init", withExtension: "csv")!

        let hiddenLayer = NeuralNetworkLayer(weights: Matrix(csvURL: hidden_URL),
                                             activationFunction: .sigmoid)
        let outputLayer = NeuralNetworkLayer(weights: Matrix(csvURL: output_URL),
                                             activationFunction: .sigmoid)

        var NN = NeuralNetwork(layers: [hiddenLayer, outputLayer])
        
        let X_all_temp = Matrix(csvURL: X_URL)
        let Y_all_temp = Matrix(csvURL: Y_URL)

        let perm = randomPermutation(X_all_temp.width)
        let X_all = X_all_temp[0..<X_all_temp.height, perm]
        let Y_all = Y_all_temp[0..<Y_all_temp.height, perm]

        let M_test = Int(Double(X_all.width) * 0.7)
        let X = X_all[0..<X_all.height, 0..<M_test]
        let Y = Y_all[0..<Y_all.height, 0..<M_test]
        let X_test = X_all[0..<X_all.height, M_test..<X_all.width]
        let Y_test = Y_all[0..<Y_all.height, M_test..<Y_all.width]
        
        let networkOptimizable = NeuralNetworkOptimizable(network: NN, X: X, Y: Y, lambda: 1)
        let cg = ConjugateGradientOptimizer(maxIterations: 1000)
        let (history, optimum) = cg.optimize(networkOptimizable)
        plot(history)
        
        NN.setAllWeights(optimum)
        
        let Yp_test = NN.forwardPropagate(X_test)
        var correct = 0
        for (yp, y) in zip(Yp_test.columns, Y_test.columns) {
            var maxValue = 0.0
            var maxIndex = -1
            
            var correctMaxValue = 0.0
            var correctMaxIndex = -1
            for i in 0..<yp.height {
                if yp[i] > maxValue {
                    maxValue = yp[i]
                    maxIndex = i
                }
                
                if y[i] > correctMaxValue {
                    correctMaxValue = y[i]
                    correctMaxIndex = i
                }
            }
            
            correct += maxIndex == correctMaxIndex ? 1 : 0
        }
        
        print("70/30 Testing set accuracy: \(Double(correct) / Double(Y_test.width))")
    }


    static var allTests : [(String, (NeuralNetworkTests) -> () throws -> Void)] {
        return [
            ("testXORLearn", testXORLearn),
            ("testMNIST", testMNIST)
        ]
    }
}
