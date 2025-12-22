using JuGNLSE
using Documenter

DocMeta.setdocmeta!(JuGNLSE, :DocTestSetup, :(using JuGNLSE); recursive=true)

makedocs(;
    modules=[JuGNLSE],
    authors="Brian Sinquin <148503669+brian-sinquin@users.noreply.github.com> and contributors",
    sitename="JuGNLSE.jl",
    format=Documenter.HTML(;
        canonical="https://brian-sinquin.github.io/JuGNLSE.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/brian-sinquin/JuGNLSE.jl", devbranch="master")
