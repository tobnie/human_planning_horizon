import matplotlib

soft_color_alpha = 0.55

CMAP = 'coolwarm'
blue = C0 = matplotlib.cm.get_cmap('coolwarm')(0.05)
red = C1 = matplotlib.cm.get_cmap('coolwarm')(0.95)
white = '#F7F7F7'

blue_kde = (C0[0], C0[1], C0[2], 0.75)
red_kde = (C1[0], C1[1], C1[2], 0.75)

blue_soft = C0_soft = (C0[0], C0[1], C0[2], soft_color_alpha)
red_soft = C1_soft = (C1[0], C1[1], C1[2], soft_color_alpha)

C0_hard = matplotlib.cm.get_cmap('coolwarm')(0.0)
C1_hard = matplotlib.cm.get_cmap('coolwarm')(1.0)

figsize = (6, 4)

font_size = 1
