import pygame


# draw some text into an area of a surface
# automatically wraps words
# returns any text that didn't get blitted
def drawText(surface, text, color, rect, font_size, aa=True, bkg=None, alignment='center'):
    font = pygame.font.SysFont(pygame.font.get_default_font(), font_size)
    rect = pygame.Rect(rect)
    y = rect.top
    lineSpacing = -2

    # get the height of the font
    fontHeight = font.size("Tg")[1]

    while text:
        i = 1

        # determine if the row of text will be outside our area
        if y + fontHeight > rect.bottom:
            break

        # determine maximum width of line
        while font.size(text[:i])[0] < rect.width and i < len(text):
            i += 1

        # if we've wrapped the text, then adjust the wrap to the last word
        if i < len(text):
            i = text.rfind(" ", 0, i) + 1

        # render the line and blit it to the surface
        if bkg:
            image = font.render(text[:i], 1, color, bkg)
            image.set_colorkey(bkg)
        else:
            image = font.render(text[:i], aa, color)

        if alignment == 'center':
            blit_x = rect.centerx - font.size(text)[0] / 2
        elif alignment == 'left':
            blit_x = rect.x
        elif alignment == 'right':
            blit_x = rect.right - font.size(text)[0]
        else:
            raise Exception("Unknown alignment: {}".format(alignment))

        surface.blit(image, (blit_x, y))
        y += fontHeight + lineSpacing

        # remove the text we just blitted
        text = text[i:]

    return text


def get_size_of_text(text, font_size):
    """ Returns the size of a text in pixels. """
    font = pygame.font.SysFont(pygame.font.get_default_font(), font_size)
    return font.size(text)
