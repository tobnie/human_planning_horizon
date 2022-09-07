import matplotlib

soft_color_alpha = 0.55

CMAP = 'coolwarm'
C0 = matplotlib.cm.get_cmap('coolwarm')(0.05)
C1 = matplotlib.cm.get_cmap('coolwarm')(0.95)

C0_soft = (C0[0], C0[1], C0[2], soft_color_alpha)
C1_soft = (C1[0], C1[1], C1[2], soft_color_alpha)

C0_hard = matplotlib.cm.get_cmap('coolwarm')(0.0)
C1_hard = matplotlib.cm.get_cmap('coolwarm')(1.0)

figsize = (6, 4)  # TODO apply to paper figures

font_size = 1
