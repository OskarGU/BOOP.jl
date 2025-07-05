using BOOP
using Documenter

DocMeta.setdocmeta!(BOOP, :DocTestSetup, :(using BOOP); recursive=true)

makedocs(;
    modules=[BOOP],
    authors="Oskar Gustafsson < oskar.gstfssn@gmail.com >",
    repo="https://github.com/OskarGU/BOOP.jl/blob/{commit}{path}#{line}", # Added this line
    sitename="BOOP.jl",
    format=Documenter.HTML(;
        canonical="https://OskarGU.github.io/BOOP.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/OskarGU/BOOP.jl",
    devbranch="main",
)
