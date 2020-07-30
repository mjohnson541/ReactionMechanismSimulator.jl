using PyCall
println("Building ReactionMechanismSimulator")
if PyCall.pyversion.major != 3 || PyCall.pyversion.minor != 7
    using Conda
    using Pkg
    Conda.add("python==3.7")
    Pkg.build("PyCall")
end