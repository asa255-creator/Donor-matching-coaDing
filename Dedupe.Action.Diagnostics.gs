/** Action handlers split for safer per-area changes */

function diagnostic_analyzeMatching() {
  const ss = SpreadsheetApp.getActive();

  // Validate required sheets exist
  const krefSheet = ss.getSheetByName('KREF_Exports');
  const fecSheet = ss.getSheetByName('FEC_Exports');

  if (!krefSheet || !fecSheet) {
    SpreadsheetApp.getUi().alert('KREF_Exports or FEC_Exports sheet not found. Please ensure your data is loaded.');
    return;
  }

  // Get web app URL
  const webAppUrl = dl_getExecUrlFromOptions_();
  if (!webAppUrl) {
    SpreadsheetApp.getUi().alert(
      'Put your Web App URL ending with /exec in Options!I2.\nExample:\nhttps://script.google.com/macros/s/AKfycb.../exec'
    );
    return;
  }

  // Read match threshold from Options!J2
  const optionsSheet = ss.getSheetByName('Options');
  const threshold = optionsSheet ? Number(optionsSheet.getRange('J2').getValue() || 0.7) : 0.7;

  // Export sheets to Drive as CSV
  const krefData = krefSheet.getDataRange().getValues();
  const fecData = fecSheet.getDataRange().getValues();

  const krefFile = cd_createTempCsv_(krefData, 'kref_export');
  const fecFile = cd_createTempCsv_(fecData, 'fec_export');

  // Make files publicly readable (temporary) for direct download
  krefFile.setSharing(DriveApp.Access.ANYONE_WITH_LINK, DriveApp.Permission.VIEW);
  fecFile.setSharing(DriveApp.Access.ANYONE_WITH_LINK, DriveApp.Permission.VIEW);

  // Get direct download URLs
  const krefUrl = 'https://drive.google.com/uc?export=download&id=' + krefFile.getId();
  const fecUrl = 'https://drive.google.com/uc?export=download&id=' + fecFile.getId();

  // Create token
  const token = dl_makeToken_();
  const until = Date.now() + 30 * 60 * 1000; // 30 minutes

  // Create diagnostic job (same pattern as training)
  const job = {
    krefUrl: webAppUrl + '?csv=kref&token=' + encodeURIComponent(token),
    fecUrl: webAppUrl + '?csv=fec&token=' + encodeURIComponent(token),
    modelUrl: webAppUrl + '?model=1'
  };

  // Store file IDs, token, and job (use dl_ prefix for compatibility with existing endpoints)
  const props = PropertiesService.getDocumentProperties();
  props.setProperty('dl_token', token);
  props.setProperty('dl_token_until', String(until));
  props.setProperty('dl_kref_fileId', krefFile.getId());
  props.setProperty('dl_fec_fileId', fecFile.getId());
  dl_setTokenCsvFileIds_(token, krefFile.getId(), fecFile.getId());
  props.setProperty('diag_job_json', JSON.stringify(job));

  // Build command using bundle (same pattern as training)
  const fullCmd =
    "curl -sSL '" + webAppUrl + "?diagnostic=1' | " +
    "python3 - --bundle '" + webAppUrl + "?diag_job=1&token=" + token + "' " +
    "--threshold " + threshold;

  // Show command in dialog
  const html = HtmlService.createHtmlOutput(
    '<div style="font-family:monospace; padding:20px; word-wrap:break-word;">' +
    '<h3>🔍 Matching Quality Diagnostic</h3>' +
    '<p><strong>Step 1:</strong> Ensure you have the diagnostic script:</p>' +
    '<pre style="background:#f5f5f5; padding:10px; overflow-x:auto;">' +
    'cd Donor-matching-coaDing\n' +
    'git pull  # Get latest version' +
    '</pre>' +
    '<p><strong>Step 2:</strong> Copy and paste this command into Terminal:</p>' +
    '<textarea readonly style="width:100%; height:120px; font-family:monospace; font-size:11px;">' + fullCmd + '</textarea>' +
    '<p style="margin-top:20px;"><strong>What this does:</strong></p>' +
    '<ul style="font-size:13px;">' +
    '<li>Downloads KREF and FEC data from Google Drive</li>' +
    '<li>Downloads your trained model weights</li>' +
    '<li>Runs diagnostic analysis</li>' +
    '</ul>' +
    '<p><strong>Expected Output:</strong></p>' +
    '<ul style="font-size:13px;">' +
    '<li>📊 Distribution histogram (should show bimodal pattern)</li>' +
    '<li>⚠️ List of obvious misses (same address + similar names)</li>' +
    '<li>📋 Example pairs at different confidence levels</li>' +
    '<li>💡 Recommendations for improving the model</li>' +
    '</ul>' +
    '<p style="color:#666; font-size:12px;"><em>Downloads expire in 30 minutes</em></p>' +
    '</div>'
  ).setWidth(650).setHeight(550);

  SpreadsheetApp.getUi().showModalDialog(html, 'Diagnostic Analysis');
}
