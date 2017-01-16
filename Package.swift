import PackageDescription

let package = Package(
    name: "SwiftLearn",
    dependencies: [
        .Package(url: "https://github.com/donald-pinckney/SwiftNum", Version(1, 9, 8))
    ]
)
