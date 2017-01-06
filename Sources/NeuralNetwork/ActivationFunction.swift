//
//  ActivationFunction.swift
//  AES-NN
//
//  Created by Donald Pinckney on 12/27/16.
//
//

import Foundation
import Linear

public enum ActivationFunction {
    case identity // g(x) = x
    case sigmoid // g(x) = 1 / (1 + e^(-x))
    case tanh // g(x) = tanh(x)
    
    public func evaluate(_ x: Double) -> Double {
        switch self {
        case .identity:
            return x
        case .sigmoid:
            return 1 / (1 + exp(-x))
        case .tanh:
            return Foundation.tanh(x)
        }
    }
    public func evaluate(_ X: Matrix) -> Matrix {
        switch self {
        case .identity:
            return X
        case .sigmoid:
            return 1 ./ (1 + exp(-X))
        case .tanh:
            return Linear.tanh(X)
        }
    }
    
    public func evaluateDerivative(_ x: Double) -> Double {
        switch self {
        case .identity:
            return 1
        case .sigmoid:
            let s = evaluate(x)
            return s * (1 - s)
        case .tanh:
            return 1 / (cosh(x) * cosh(x))
        }
    }
    
    public func evaluateDerivative(_ X: Matrix) -> Matrix {
        switch self {
        case .identity:
            return Matrix.fill(X.height, X.width, value: 1)
        case .sigmoid:
            let s = evaluate(X)
            return s .* (1 - s)
        case .tanh:
            return 1 ./ (cosh(X) .* cosh(X))
        }
    }
}

