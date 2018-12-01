module RMS
    using PyCall
    const Chem = PyNULL()
    const molecule = PyNULL()
    const pydot = PyNULL()
    function __init__()
        copy!(Chem,pyimport_conda("rdkit.Chem","rdkit","rmg"))
        copy!(molecule,pyimport_conda("rmgpy.molecule","rmgpy","rmg"))
        copy!(pydot,pyimport_conda("pydot","pydot","rmg"))
    end
    include("Constants.jl")
    include("Tools.jl")
    include("Calculators/RateUncertainty.jl")
    include("Calculators/ThermoUncertainty.jl")
    include("Calculators/Thermo.jl")
    include("Calculators/Diffusion.jl")
    include("Calculators/Rate.jl")
    include("Calculators/Viscosity.jl")
    include("Species.jl")
    include("Solvent.jl")
    include("Reaction.jl")
    include("Phase.jl")
    include("PhaseState.jl")
    include("Interface.jl")
    include("Domain.jl")
    include("Parse.jl")
    include("Reactor.jl")
    include("Solution.jl")
    include("fluxdiagrams.jl")
    include("Equilibrium.jl")
end
