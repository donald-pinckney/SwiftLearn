import PackageDescription

let package = Package(
    name: "SwiftLearn",
    dependencies: [
        .Package(url: "../SwiftNum", Version(1, 9, 0))
    ]
)
