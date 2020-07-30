using PyCall
if PyCall.pyversion.major != 3 || PyCall.pyversion.minor != 7
    using Conda
    const Pkg = Base.require(Base.PkgId(Base.UUID("44cfe95a-1eb2-52ea-b672-e2afdf69b78f"), "Pkg"))
    Conda.add_channel("rmg")
    Conda.add_channel("defaults")
    Conda.add("python==3.7")
    Conda.clean()
    Conda.update()
    Conda.add("rmg=3.0.0")
    Conda.clean()
    Conda.update()
    Pkg.build("PyCall")
end