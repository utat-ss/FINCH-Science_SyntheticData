class Menu:
    def __init__(self, items):
        self.items = items

lst = ['Home', 'About', 'Contact']
m = Menu(lst)
print(m.items)
lst.append('Blog')
print(m.items)  # Should still print ['Home', 'About', 'Contact'], but it prints ['Home', 'About', 'Contact', 'Blog']
