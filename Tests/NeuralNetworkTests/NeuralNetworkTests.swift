import XCTest
@testable import NeuralNetwork

class NeuralNetworkTests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct results.
        XCTAssertEqual(NeuralNetwork().text, "Hello, World!")
    }


    static var allTests : [(String, (NeuralNetworkTests) -> () throws -> Void)] {
        return [
            ("testExample", testExample),
        ]
    }
}
