# make_release.ps1 — Génère piezo_kriging.zip pour le dépôt QGIS sur NAS
#
# Usage : .\make_release.ps1
# Le ZIP produit contient un seul dossier piezo_kriging/ à sa racine,
# ce qu'attend QGIS pour l'installation depuis un dépôt ou depuis un ZIP.

$ErrorActionPreference = "Stop"

# ── Configuration ──────────────────────────────────────────────
$PLUGIN_NAME = "piezo_kriging"
$NAS_IP      = "NAS_IP"                       # <-- remplacer par l'IP du NAS
$NAS_FOLDER  = "qgis-plugins"                 # dossier servi en HTTP sur le NAS
$NAS_PATH    = "\\$NAS_IP\$NAS_FOLDER"        # chemin UNC du partage réseau

$FILES = @(
    "__init__.py",
    "kriging_engine.py",
    "piezo_kriging.py",
    "piezo_dialog.py",
    "metadata.txt",
    "exemple_donnees.csv",
    "README.md",
    "icons"
)
# ───────────────────────────────────────────────────────────────

$ROOT   = $PSScriptRoot
$TEMP   = Join-Path $env:TEMP "piezo_release_tmp"
$PLUGIN = Join-Path $TEMP $PLUGIN_NAME
$ZIP    = Join-Path $ROOT "$PLUGIN_NAME.zip"

# Lire la version dans metadata.txt
$version = (Select-String -Path (Join-Path $ROOT "metadata.txt") -Pattern "^version\s*=\s*(.+)").Matches[0].Groups[1].Value.Trim()
Write-Host "Version : $version"

# Nettoyer et recréer le répertoire temporaire
if (Test-Path $TEMP) { Remove-Item $TEMP -Recurse -Force }
New-Item -ItemType Directory -Path $PLUGIN | Out-Null

# Copier les fichiers du plugin
foreach ($f in $FILES) {
    $src = Join-Path $ROOT $f
    if (Test-Path $src) {
        Copy-Item $src -Destination $PLUGIN -Recurse -Force
    } else {
        Write-Warning "Fichier absent (ignoré) : $f"
    }
}

# Créer le ZIP (structure : piezo_kriging/ à la racine)
if (Test-Path $ZIP) { Remove-Item $ZIP -Force }
Compress-Archive -Path "$TEMP\*" -DestinationPath $ZIP
Write-Host "ZIP créé : $ZIP"

# Nettoyer le dossier temporaire
Remove-Item $TEMP -Recurse -Force

# ── Déploiement sur le NAS ──────────────────────────────────────
if (Test-Path $NAS_PATH) {
    # Copier le ZIP
    Copy-Item $ZIP -Destination $NAS_PATH -Force
    Write-Host "ZIP déployé sur : $NAS_PATH"

    # Mettre à jour la version dans plugins.xml sur le NAS
    $xmlNas = Join-Path $NAS_PATH "plugins.xml"
    $xmlLocal = Join-Path $ROOT "plugins.xml"
    if (Test-Path $xmlLocal) {
        # Mettre à jour le numéro de version dans le XML local puis copier
        $xml = [xml](Get-Content $xmlLocal -Encoding UTF8)
        $node = $xml.plugins.pyqgis_plugin
        $node.SetAttribute("version", $version)
        $node.version = $version
        $xml.Save($xmlLocal)
        Copy-Item $xmlLocal -Destination $NAS_PATH -Force
        Write-Host "plugins.xml mis à jour (version $version) et déployé sur : $NAS_PATH"
    }
} else {
    Write-Warning "NAS non accessible ($NAS_PATH). Copiez manuellement :"
    Write-Warning "  - $ZIP"
    Write-Warning "  - $(Join-Path $ROOT 'plugins.xml')"
    Write-Warning "vers le dossier partagé du NAS."
}

Write-Host ""
Write-Host "Terminé. URL du dépôt à ajouter dans QGIS :"
Write-Host "  http://$NAS_IP/$NAS_FOLDER/plugins.xml"
