import XCTest
import MLX
import MLXRandom

final class TestSlots: XCTestCase {
    func testSlots() {
        let slotPerTokenArr: [Int32] = [0, 1, 2, 3, 4, 5, 6, 7]
        let slotPerToken = MLXArray(slotPerTokenArr).asType(.uint32)
        let slots = slotPerToken.asArray(Int32.self)
        print(slots)
    }
}
