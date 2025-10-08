import pycolmap

# Afficher tous les attributs du module
print(dir(pycolmap))

# Pour plus de lisibilité, vous pouvez les afficher un par ligne
for attr in dir(pycolmap):
    print(attr)

# Pour filtrer uniquement les attributs qui ne commencent pas par '_' (attributs publics)
public_attrs = [attr for attr in dir(pycolmap) if not attr.startswith('_')]
for attr in public_attrs:
    print(attr)

# Pour obtenir de l'aide sur un attribut spécifique
help(pycolmap.extract_features)  # Remplacez par l'attribut qui vous intéresse

print(pycolmap.__version__)
