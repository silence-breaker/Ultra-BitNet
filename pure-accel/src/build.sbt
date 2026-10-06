ThisBuild / scalaVersion := "2.12.18"
ThisBuild / organization := "org.ultrabitnet"
ThisBuild / version := "0.1.0"

val spinalVersion = "1.11.0"

lazy val accelerator = (project in file("."))
  .settings(
    name := "ultra-bitnet-accelerator",
    Compile / scalaSource := baseDirectory.value / "main" / "scala",
    scalacOptions ++= Seq("-deprecation", "-feature", "-unchecked", "-target:jvm-1.8"),
    libraryDependencies ++= Seq(
      "com.github.spinalhdl" %% "spinalhdl-core" % spinalVersion,
      "com.github.spinalhdl" %% "spinalhdl-lib" % spinalVersion,
      compilerPlugin("com.github.spinalhdl" %% "spinalhdl-idsl-plugin" % spinalVersion)
    ),
    Compile / run / fork := true,
    Compile / run / javaOptions ++= Seq("-Xmx12G")
  )

addCommandAlias("generateBoardRtl", "runMain ultrabitnet.accel.GenerateBitNetResidentBoardAccelerator ../build/rtl")
