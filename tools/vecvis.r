#install.packages("matlib")

library(matlib)
library(rgl)

par3d(windowRect = c(20, 30, 800, 800))
#open3d()

draw_vec <- function(vec, color, orig = c(0, 0, 0)) {
  vectors3d(
    vec,
    origin = orig,
    headlength = 0.035,
    ref.length = NULL,
    color = color,
    radius = 1/60,
    labels = TRUE,
    cex.lab = 1.2,
    adj.lab = 0.5,
    frac.lab = 1.1,
    draw = TRUE,
  )
}

wo_dir = c(-0.415600, 0.881494, 0.224153)
wi_dir=c(-0.580490, 0.809596, -0.087099)
normal_dir=c(-0.999731, -0.019059,-0.013196)
geom_normal_dir=c(-0.848574, -0.434451, -0.301950)
wi_offset=c(-0.000013, -0.000013, -0.000009)
last_hit_pos=c(-0.000096, 0.000162, 0.000035)
its_pos=c(0.565741, 1.960749, -1.756458)

wo <- matrix(wo_dir, nrow = 1, ncol = 3)
rownames(wo) <- c("wo")

wi <- matrix(wi_dir, nrow = 1, ncol = 3)
rownames(wi) <- c("wi")

normal <- matrix(normal_dir, nrow = 1, ncol = 3)
rownames(normal) <- c("n")

geom_normal <- matrix(geom_normal_dir, nrow = 1, ncol = 3)
rownames(geom_normal) <- c("geom_n")

planes3d(normal[1, 1], normal[1, 2], normal[1, 3], 0, alpha = 0.7, color = "grey")
points3d(wi_offset, color = "magenta")
points3d(last_hit_pos, color = "green")
points3d(c(0, 0, 0))

# Hack for having wider plane
points3d(c(-1, -1, -1))
points3d(c(1, 1, 1))

draw_vec(wo, "blue")
draw_vec(wi, "orange", orig = wi_offset)
draw_vec(normal, "red")
draw_vec(geom_normal, "red")

rgl.bringtotop()

Sys.sleep(100)
