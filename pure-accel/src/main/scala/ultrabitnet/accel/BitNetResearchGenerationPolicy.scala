package ultrabitnet.accel

/**
  * Fail-closed opt-in shared by retained BTN1/BTP1/GEMV research generators.
  *
  * These generators remain useful for numerical and board-baseline regression,
  * but their historical names must not make them look like production build
  * entries.  Requiring a literal command-line switch makes every invocation
  * deliberate and keeps automation from selecting them by accident.
  */
object BitNetResearchGenerationPolicy {
  val OptIn: String = "--allow-legacy-research"

  def requireOptIn(entry: String, args: Array[String]): Array[String] = {
    require(
      args.contains(OptIn),
      s"$entry is a retained BTN1/BTP1/GEMV research generator, not a " +
        "production entry. Use " +
        "ultrabitnet.accel.GenerateBitNetResidentBoardAccelerator for production; " +
        s"pass $OptIn only for an intentional legacy regression."
    )
    args.filterNot(_ == OptIn)
  }
}
