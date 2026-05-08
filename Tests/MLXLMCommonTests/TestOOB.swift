import XCTest
import MLX
import MLXRandom

final class TestOOB: XCTestCase {
    func testOOB() {
        MLXRandom.seed(0)
        let x = MLXArray([[1.0, 2.0]]) // [1, 2]
        let order = MLXArray([0, 1, 2]) // [3]
        let y = x[order]
        print(y)
        MLX.eval(y)
        print(y)
    }
}
