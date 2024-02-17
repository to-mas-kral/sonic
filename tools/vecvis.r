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

normal_shade_dir = c(0.000000, 0.000000, 1.000000)
wo_shade_dir = c(0.006536, -0.182330, 0.983216)
wm_shade_dir = c(-0.036927, -0.464259, 0.884929)
wo_dir = c(-0.091345, -0.979384, -0.180174)
wi_dir = c(-0.076787, -0.701959, -0.708066)
normal_dir = c(-0.121888, -0.992544, 0.000000)
spawn_normal_dir = c(-0.121888, -0.992544, 0.000000)
wm_dir = c(-0.088075, -0.880761, -0.465298)

normal_shade <- matrix(normal_shade_dir, nrow = 1, ncol = 3)
rownames(normal_shade) <- c("normal_shade")
draw_vec(normal_shade, "red")

wo_shade <- matrix(wo_shade_dir, nrow = 1, ncol = 3)
rownames(wo_shade) <- c("wo_shade")
draw_vec(wo_shade, "blue")

wm_shade <- matrix(wm_shade_dir, nrow = 1, ncol = 3)
rownames(wm_shade) <- c("wm_shade")
draw_vec(wm_shade, "orange")

wo <- matrix(wo_dir, nrow = 1, ncol = 3)
rownames(wo) <- c("wo")

wi <- matrix(wi_dir, nrow = 1, ncol = 3)
rownames(wi) <- c("wi")

wm <- matrix(wm_dir, nrow = 1, ncol = 3)
rownames(wm) <- c("wm")

normal <- matrix(normal_dir, nrow = 1, ncol = 3)
rownames(normal) <- c("n")

spawn_normal <- matrix(spawn_normal_dir, nrow = 1, ncol = 3)
rownames(spawn_normal) <- c("geom_n")

planes3d(normal[1, 1], normal[1, 2], normal[1, 3], 0, alpha = 0.7, color = "grey")
points3d(c(0, 0, 0))

# Hack for having wider plane
points3d(c(-1, -1, -1))
points3d(c(1, 1, 1))

draw_vec(wi, "yellow")
draw_vec(wo, "blue")
draw_vec(normal, "red")
#draw_vec(spawn_normal, "red")
draw_vec(wm, "orange")

rgl.bringtotop()

Sys.sleep(100)
