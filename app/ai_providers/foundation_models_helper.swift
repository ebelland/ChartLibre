// ChartLibre's bridge to Apple's on-device model: JSON lines on stdin and stdout, one process a conversation.
//
//   in:  start {instructions, tools}, user {turn, text}, tool_result {turn, content, is_error}, cancel
//   out: text {turn, text}, tool_call {turn, name, input}, done {turn}, error {turn, kind, message}
//
// The framework runs a tool inside its response: the tool writes tool_call and waits for its result, one at a
// time, so the application's confirmations and Undo stay on its own side. `--check` says whether the model can
// be used, and exits. Ends when stdin closes.
import Foundation
import FoundationModels

let outLock = NSLock()

func emit(_ object: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: object) else { return }
    outLock.lock()
    FileHandle.standardOutput.write(data + Data([10]))
    outLock.unlock()
}

func unavailableReason() -> String {
    switch SystemLanguageModel.default.availability {
    case .available: return ""
    case .unavailable(.deviceNotEligible): return "device_not_eligible"
    case .unavailable(.appleIntelligenceNotEnabled): return "apple_intelligence_off"
    case .unavailable(.modelNotReady): return "model_not_ready"
    default: return "unavailable"
    }
}

/// A JSON schema's node as the framework's schema. Anything it cannot say is asked for as text.
func dynamicSchema(_ node: [String: Any], name: String) -> DynamicGenerationSchema {
    if let choices = node["enum"] as? [String] {
        return DynamicGenerationSchema(name: name, anyOf: choices)
    }
    switch node["type"] as? String {
    case "number": return DynamicGenerationSchema(type: Double.self)
    case "integer": return DynamicGenerationSchema(type: Int.self)
    case "boolean": return DynamicGenerationSchema(type: Bool.self)
    case "array":
        let items = node["items"] as? [String: Any] ?? [:]
        return DynamicGenerationSchema(arrayOf: dynamicSchema(items, name: name + "_item"))
    case "object":
        let properties = node["properties"] as? [String: Any] ?? [:]
        let required = Set(node["required"] as? [String] ?? [])
        return DynamicGenerationSchema(name: name, properties: properties.keys.sorted().map { key in
            let child = properties[key] as? [String: Any] ?? [:]
            return DynamicGenerationSchema.Property(
                name: key, description: child["description"] as? String,
                schema: dynamicSchema(child, name: name + "_" + key), isOptional: !required.contains(key))
        })
    default: return DynamicGenerationSchema(type: String.self)
    }
}

final class Bridge: @unchecked Sendable {
    private let lock = NSLock()
    private var waiting: CheckedContinuation<String, Error>?
    private var turn = 0
    private var task: Task<Void, Never>?
    private var session: LanguageModelSession?
    private let gate = Gate()

    func start(instructions: String, tools: [[String: Any]]) throws {
        let bridged = try tools.map { spec -> BridgeTool in
            let name = spec["name"] as? String ?? ""
            let root = dynamicSchema(spec["schema"] as? [String: Any] ?? ["type": "object"], name: name + "_input")
            return BridgeTool(name: name, description: spec["description"] as? String ?? "",
                              parameters: try GenerationSchema(root: root, dependencies: []), bridge: self)
        }
        session = LanguageModelSession(tools: bridged, instructions: instructions)
    }

    func ask(turn newTurn: Int, text: String) {
        cancel()
        lock.lock()
        turn = newTurn
        let session = self.session
        lock.unlock()
        guard let session else { return }
        task = Task {
            var sent = ""
            do {
                for try await partial in session.streamResponse(to: text) {
                    let whole = partial.content
                    let fresh = whole.hasPrefix(sent) ? String(whole.dropFirst(sent.count)) : whole
                    sent = whole
                    if !fresh.isEmpty && !Task.isCancelled { emit(["type": "text", "turn": newTurn, "text": fresh]) }
                }
                emit(["type": "done", "turn": newTurn])
            } catch is CancellationError {
                return
            } catch let error as LanguageModelSession.GenerationError {
                let kind: String
                switch error {
                case .exceededContextWindowSize: kind = "context"
                case .guardrailViolation, .refusal: kind = "refused"
                default: kind = "other"
                }
                emit(["type": "error", "turn": newTurn, "kind": kind, "message": error.localizedDescription])
            } catch {
                emit(["type": "error", "turn": newTurn, "kind": "other", "message": error.localizedDescription])
            }
        }
    }

    func cancel() {
        lock.lock()
        let pending = waiting
        waiting = nil
        lock.unlock()
        task?.cancel()
        pending?.resume(throwing: CancellationError())
    }

    func result(content: String, isError: Bool) {
        lock.lock()
        let pending = waiting
        waiting = nil
        lock.unlock()
        pending?.resume(returning: isError ? "Error: " + content : content)
    }

    /// One call at a time: the application answers one tool_call before it sees the next.
    func callTool(name: String, input: String) async throws -> String {
        await gate.enter()
        defer { Task { await gate.leave() } }
        return try await withCheckedThrowingContinuation { continuation in
            lock.lock()
            waiting = continuation
            let current = turn
            lock.unlock()
            let parsed = (try? JSONSerialization.jsonObject(with: Data(input.utf8))) ?? [String: Any]()
            emit(["type": "tool_call", "turn": current, "name": name, "input": parsed])
        }
    }
}

actor Gate {
    private var busy = false
    private var queue: [CheckedContinuation<Void, Never>] = []

    func enter() async {
        if busy { await withCheckedContinuation { queue.append($0) } } else { busy = true }
    }

    func leave() {
        if queue.isEmpty { busy = false } else { queue.removeFirst().resume() }
    }
}

struct BridgeTool: Tool {
    typealias Arguments = GeneratedContent
    typealias Output = String
    let name: String
    let description: String
    let parameters: GenerationSchema
    let bridge: Bridge

    func call(arguments: GeneratedContent) async throws -> String {
        try await bridge.callTool(name: name, input: arguments.jsonString)
    }
}

@main
struct Helper {
    static func main() async {
        if CommandLine.arguments.contains("--check") {
            print(unavailableReason())
            return
        }
        let bridge = Bridge()
        let reader = Thread {
            while let line = readLine(strippingNewline: true) {
                guard let message = (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any]
                else { continue }
                switch message["type"] as? String {
                case "start":
                    do {
                        try bridge.start(instructions: message["instructions"] as? String ?? "",
                                         tools: message["tools"] as? [[String: Any]] ?? [])
                    } catch {
                        emit(["type": "error", "turn": 0, "kind": "other", "message": "\(error)"])
                    }
                case "user":
                    bridge.ask(turn: message["turn"] as? Int ?? 0, text: message["text"] as? String ?? "")
                case "tool_result":
                    bridge.result(content: message["content"] as? String ?? "",
                                  isError: message["is_error"] as? Bool ?? false)
                case "cancel": bridge.cancel()
                default: break
                }
            }
            exit(0)
        }
        reader.start()
        while true { try? await Task.sleep(nanoseconds: 3_600_000_000_000) }
    }
}
