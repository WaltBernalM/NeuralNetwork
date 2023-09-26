import Foundation

enum DataSizeType: Int {
    case oneD = 1
    case twoD
    case threeD
}
struct DataSize {
    var type: DataSizeType
    var width: Int
    var height: Int?
    var depth: Int?
    init(width: Int) {
        type = .oneD
        self.width = width
    }
    init(width: Int, height: Int) {
        type = .twoD
        self.width = width
        self.height = height
    }
    init(width: Int, height: Int, depth: Int) {
        type = .threeD
        self.width = width
        self.height = height
        self.depth = depth
    }
}
struct DataPiece {
    var size: DataSize
    var body: [Float]
    func get(x: Int) -> Float {
        return body[x]
    }
    func get(x: Int, y: Int) -> Float {
        return body[x+y*size.width]
    }
    func get(x: Int, y: Int, z: Int) -> Float {
        return body[z+(x+y*size.width)*size.depth!]
    }
}
struct DataSample {
    var input: DataPiece
    var output: DataPiece
}
struct Dataset {
    var items: [DataSample]
}

final class NeuralNetwork {
    var layers: [Layer] = []
    var learningRate = Float(0.05)
    var epochs = 30
    var batchSize = 16
    func deltaWeights(row: DataPiece) {
        var input = row
        for i in 0..<layers.count {
            input = layers[i].deltaWeights(input: input, learningRate: learningRate)
        }
    }
    func forward(networkInput: DataPiece) -> DataPiece {
        var input = networkInput
        for i in 0..<layers.count {
            input = layers[i].forward(input: input)
        }
        return input
    }
    func backward(expected: DataPiece) {
        var input = expected
        var previous: Layer? = nil
        for i in (0..<layers.count).reversed() {
            input = layers[i].backward(input: input, previous: previous)
            previous = layers[i]
        }
    }
    func train(set: Dataset) -> Float {
        var error = Float.zero
        for epoch in 0..<epochs {
            var shuffledSet = set.items.shuffled()
            error = Float.zero
            while !shuffledSet.isEmpty {
                let batch = shuffledSet.prefix(batchSize)
                for item in batch {
                    let predictions = forward(networkInput: item.input)
                    for i in 0..<item.output.body.count {
                        error+=pow(item.output.body[i]-predictions.body[i], 2)/2
                    }
                    backward(expected: item.output)
                    deltaWeights(row: item.input)
                }
                for layer in layers {
                    layer.updateWeights()
                }
                shuffledSet.removeFirst(min(batchSize,shuffledSet.count))
            }
            print("Epoch \(epoch+1), error \(error).")
        }
        return error
    }
    func predict(input: DataPiece) -> [Float] {
        return forward(networkInput: input).body
    }
}
class Layer {
    var neurons: [Neuron] = []
    var function: ActivationFunction
    var output: DataPiece?
    init(function: ActivationFunction) {
        self.function = function
    }
    func updateWeights() {
        let neuronsCount = neurons.count
        neurons.withUnsafeMutableBufferPointer { neuronsPtr in
            DispatchQueue.concurrentPerform(iterations: neuronsCount, execute: { i in
                neuronsPtr[i].weights.withUnsafeMutableBufferPointer { weightsPtr in
                    neuronsPtr[i].weightsDelta.withUnsafeMutableBufferPointer { deltaPtr in
                      let weightsCount = deltaPtr.count
                        DispatchQueue.concurrentPerform(iterations: weightsCount, execute: { j in
                            weightsPtr[j] += deltaPtr[j]
                            deltaPtr[j] = 0
                        })
                    }
                }
            })
        }
    }
    func deltaWeights(input: DataPiece, learningRate: Float) -> DataPiece {
    neurons.withUnsafeMutableBufferPointer { neuronsPtr in
        input.body.withUnsafeBufferPointer { inputPtr in
            DispatchQueue.concurrentPerform(iterations: neuronsPtr.count, execute: { i in
                neuronsPtr[i].weightsDelta.withUnsafeMutableBufferPointer { deltaPtr in
                    DispatchQueue.concurrentPerform(iterations: deltaPtr.count, execute: { j in
                        deltaPtr[j] += learningRate * neuronsPtr[i].delta * inputPtr[j]
                    })
                    neuronsPtr[i].bias += learningRate * neuronsPtr[i].delta
                }
            })
        }
    }
    return output!
}
    func forward(input: DataPiece) -> DataPiece {
        input.body.withUnsafeBufferPointer { inputPtr in
            output?.body.withUnsafeMutableBufferPointer { outputPtr in
                neurons.withUnsafeBufferPointer { neuronsPtr in
                    DispatchQueue.concurrentPerform(iterations: neuronsPtr.count, execute: { i in
                        var out = neuronsPtr[i].bias
                        neuronsPtr[i].weights.withUnsafeBufferPointer { weightsPtr in
                            DispatchQueue.concurrentPerform(iterations: neuronsPtr[i].weights.count, execute: { i in
                                out += weightsPtr[i] * inputPtr[i]
                            })
                        }
                        outputPtr[i] = function.transfer(input: out)
                    })
                }
            }
        }
        return output!
    }
    func backward(input: DataPiece, previous: Layer?) -> DataPiece {
        var errors = Array(repeating: Float.zero, count: neurons.count)
        if let previous = previous {
            for j in 0..<neurons.count {
                for neuron in previous.neurons {
                    errors[j] += neuron.weights[j]*neuron.delta
                }
            }
        } else {
            for j in 0..<neurons.count {
                errors[j] = input.body[j] - output!.body[j]
            }
        }
        for j in 0..<neurons.count {
            neurons[j].delta = errors[j] * function.derivative(output: output!.body[j])
        }
        return output!
    }
}
struct Neuron {
    var weights: [Float]
    var weightsDelta: [Float]
    var bias: Float
    var delta: Float
}
protocol ActivationFunction {
    var rawValue: Int { get }
    func transfer(input: Float) -> Float
    func derivative(output: Float) -> Float
}

struct Sigmoid: ActivationFunction {
    var rawValue: Int = 0
    func transfer(input: Float) -> Float {
        return 1.0/(1.0+exp(-input))
    }
    func derivative(output: Float) -> Float {
        return output*(1.0-output)
    }
}

func getActivationFunction(rawValue: Int) -> ActivationFunction {
    switch rawValue {
    default:
        return Sigmoid()
    }
}
enum ActivationFunctionRaw: Int {
    case sigmoid = 0
}

class Dense: Layer {
    init(inputSize: Int, neuronsCount: Int, functionRaw: ActivationFunctionRaw) {
        let function = getActivationFunction(rawValue: functionRaw.rawValue)
        super.init(function: function)
        output = .init(size: .init(width: neuronsCount), body: Array(repeating: Float.zero, count: neuronsCount))
        self.neurons = Array(repeating: Neuron(weights: [], weightsDelta: .init(repeating: Float.zero, count: inputSize), bias: 0.0, delta: 0.0), count: neuronsCount)
        for i in 0..<neuronsCount {
            var weights = [Float]()
            for _ in 0..<inputSize {
                weights.append(Float.random(in: -1.0 ... 1.0))
            }
            neurons[i].weights = weights
        }
    }
}

var network = NeuralNetwork()
network.learningRate = 0.5
network.epochs = 1000
network.batchSize = 8
network.layers = [
    Dense(inputSize: 4, neuronsCount: 4, functionRaw: .sigmoid),
    Dense(inputSize: 4, neuronsCount: 4, functionRaw: .sigmoid),
    Dense(inputSize: 4, neuronsCount: 1, functionRaw: .sigmoid)
]
let set = Dataset(items: [
    .init(input: .init(size: .init(width: 4), body: [0.0, 0.0, 0.0, 1.0]), output: .init(size: .init(width: 1), body: [0.0])),
    .init(input: .init(size: .init(width: 4), body: [1.0, 0.0, 0.0, 1.0]), output: .init(size: .init(width: 1), body: [0.0])),
    .init(input: .init(size: .init(width: 4), body: [0.0, 1.0, 1.0, 1.0]), output: .init(size: .init(width: 1), body: [0.0])),
    .init(input: .init(size: .init(width: 4), body: [0.0, 0.0, 1.0, 0.0]), output: .init(size: .init(width: 1), body: [1.0])),
    .init(input: .init(size: .init(width: 4), body: [0.0, 1.0, 0.0, 0.0]), output: .init(size: .init(width: 1), body: [1.0])),
    .init(input: .init(size: .init(width: 4), body: [1.0, 0.0, 0.0, 0.0]), output: .init(size: .init(width: 1), body: [1.0]))
])

network.train(set: set)

func makePrediction(x0: Float, x1: Float, x2: Float, x3: Float) -> Bool {
    let bodyInput = [x0, x1, x2, x3]
    let prediction = network.predict(input: .init(size: .init(width: bodyInput.count), body: bodyInput))
    print("Prediction for \(bodyInput): \(prediction)")
    return prediction[0] > 0.5
}

makePrediction(x0: 1, x1: 1, x2: 1, x3: 1)

