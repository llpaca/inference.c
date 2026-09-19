/*
 * Main.kt - CLI dispatcher for all model engines.
 *
 * Each model family lives in its own <name>_engine-style file
 * (see LlamaEngine.kt) exposing a `run<Name>(args)` entry point.
 * This mirrors how inference.c ships llama.c / llama.8b.c as separate
 * compiled programs sharing the same core primitives -- here they share
 * SafeTensors.kt / Tokenizer.kt / Kernels.kt and are chosen by subcommand
 * instead of by which binary you invoke.
 */

fun main(rawArgs: Array<String>) {
    if (rawArgs.isEmpty()) {
        printUsage()
        return
    }

    val command = rawArgs[0]
    val rest = rawArgs.drop(1)

    try {
        when (command) {
            "llama" -> runLlama(rest)
            "-h", "--help", "help" -> printUsage()
            else -> {
                System.err.println("Unknown model: $command")
                printUsage()
            }
        }
    } catch (e: IllegalStateException) {
        System.err.println("[error] ${e.message}")
        kotlin.system.exitProcess(1)
    } catch (e: java.io.IOException) {
        System.err.println("[error] ${e.message}")
        kotlin.system.exitProcess(1)
    }
}

private fun printUsage() {
    println(
        """
        inference.kt - Kotlin/JVM LLM inference engine

        Usage:
          java -jar inference.jar <model> <model_dir> "<prompt>" [maxTokens] [temperature]

        Models:
          llama    Llama 3.2 1B / 3B (Instruct or base)
                   java -jar inference.jar llama ./models/llama-3.2-1b "Tell me a joke" 200 0.0
                   java -jar inference.jar llama ./models/llama-3.2-3b "Tell me a joke" 200 0.7 --size 3b

        More models land as their own *_engine.kt file + subcommand here
        (see README.md for the roadmap: ternary/1-bit models for on-device use).
        """.trimIndent()
    )
}
