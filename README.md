# Dépannage — Erreur de tension sur chargeur de batterie LiPo

## Diagnostic initial

En cas d'erreur de tension lors de la mise en charge, utiliser la **fonction "Meter"** du chargeur pour vérifier la tension de chaque cellule individuellement.

### Critères d'évaluation

| Situation | Action |
|---|---|
| Déséquilibre entre cellules **> 1 V** | Jeter la batterie |
| Une cellule affiche **0 V** | Jeter la batterie |
| Pas de déséquilibre significatif | Procéder à la récupération (voir ci-dessous) |

> ⚠️ **Avertissement** : Une batterie LiPo endommagée peut être dangereuse (gonflement, incendie). Ne jamais ignorer une cellule à 0 V ou un fort déséquilibre.

---

## Procédure de récupération (alimentation de laboratoire)

1. Allumer l'alimentation de laboratoire
2. Régler la tension sur **12 V** (maximum **12,5 V**)
3. Connecter le **câble rouge** sur la borne positive (`+`) de la batterie
4. Connecter le **câble noir** sur la borne négative (`-`) de la batterie
5. Attendre que le courant redescende à **0 A**

La batterie est alors hors de l'état de protection et peut être rechargée normalement.

---

> 📝 *Cette procédure s'applique uniquement aux batteries LiPo présentant une erreur de tension sans déséquilibre critique entre cellules.*