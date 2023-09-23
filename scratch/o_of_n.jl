using Gadfly, Cairo

x = 0:2:32
y_linear = x .* 1.2 .+ 10
y_quadratic = x .* x .* 0.2 .+ 10

p = plot(layer(x=x,
        y=y_linear,
        Geom.line,
        Theme(default_color=colorant"forestgreen")
    ),
    layer(x=x,
        y=y_quadratic,
        Geom.line,
        Theme(default_color=colorant"navyblue")
    ),
    Guide.XLabel("Context Length (Time Steps)"),
    Guide.YLabel("Compute Needed"),
    Guide.Title("Linear vs Quadratic [O(n) vs O(n^2)]"),
    Guide.manual_color_key("Legend", ["Linear", "Quadratic"], ["forestgreen", "navyblue"]),
    Guide.xticks(label=false),
    Guide.yticks(label=false),
)
p

draw(PNG("./images/o_of_n.png", 6inch, 4inch), p)




##
points = DataFrame(index=rand(0:10, 30), val=rand(1:10, 30))
line = DataFrame(val=rand(1:10, 11), index=collect(0:10))
pointLayer = layer(points, x="index", y="val", Geom.point, Theme(default_color=colorant"green"))
lineLayer = layer(line, x="index", y="val", Geom.line)
plot(pointLayer, lineLayer, Guide.manual_color_key("Legend", ["Points", "Line"], ["green", "deepskyblue"]))