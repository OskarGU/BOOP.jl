using BOOP
using Documenter

DocMeta.setdocmeta!(BOOP, :DocTestSetup, :(using BOOP); recursive=true)

makedocs(;
    modules=[BOOP],
    authors="Oskar Gustafsson",
    sitename="BOOP.jl",
    format=Documenter.HTML(;
        canonical="https://OskarGU.github.io/BOOP.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/OskarGU/BOOP.jl",
    devbranch="master",
)
