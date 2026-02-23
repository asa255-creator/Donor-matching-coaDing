/** Action handlers split for safer per-area changes */

function cd_prepareMatchingJob_() {
  const ss = SpreadsheetApp.getActive();
  
  // Validate required sheets exist
  const mergeSheet = ss.getSheetByName('Merge output');
  const cdSheet = ss.getSheetByName('CD_donors');
  
  if (!mergeSheet) {
    SpreadsheetApp.getUi().alert('Merge output sheet not found. Please run the donor matching first.');
    return;
  }
  
  if (!cdSheet) {
    SpreadsheetApp.getUi().alert('CD_donors sheet not found. Please upload a Campaign Deputy export first.');
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
  
  // Export sheets to Drive as CSV
  const mergeData = mergeSheet.getDataRange().getValues();
  const cdData = cdSheet.getDataRange().getValues();

  const mergeFile = cd_createTempCsv_(mergeData, 'merge_output');
  const cdFile = cd_createTempCsv_(cdData, 'cd_donors');

  // Make files publicly readable (temporary) for direct download
  mergeFile.setSharing(DriveApp.Access.ANYONE_WITH_LINK, DriveApp.Permission.VIEW);
  cdFile.setSharing(DriveApp.Access.ANYONE_WITH_LINK, DriveApp.Permission.VIEW);

  // Get direct download URLs (bypassing the web app middleman)
  const mergeUrl = 'https://drive.google.com/uc?export=download&id=' + mergeFile.getId();
  const cdUrl = 'https://drive.google.com/uc?export=download&id=' + cdFile.getId();

  // Create token
  const token = dl_makeToken_();
  const until = Date.now() + 60 * 60 * 1000; // 1 hour

  // Store file IDs and token (for cleanup later)
  const props = PropertiesService.getDocumentProperties();
  props.setProperty('cd_token', token);
  props.setProperty('cd_token_until', String(until));
  props.setProperty('cd_merge_fileId', mergeFile.getId());
  props.setProperty('cd_cd_fileId', cdFile.getId());

  // Build terminal command - downloads DIRECTLY from Drive, not through web app
  const cmd =
    "curl -sSL '" + webAppUrl + "?cd_matcher=1' | " +
    "python3 - --merge '" + mergeUrl + "' " +
    "--cd '" + cdUrl + "' " +
    "--result '" + webAppUrl + "?cd_result=1&token=" + token + "' " +
    "--model '" + webAppUrl + "?model=1'";
  
  // Show command in dialog
  const html = HtmlService.createHtmlOutput(
    '<div style="font-family:monospace; padding:20px; word-wrap:break-word;">' +
    '<h3>Campaign Deputy Matching</h3>' +
    '<p>Copy and paste this command into Terminal:</p>' +
    '<textarea readonly style="width:100%; height:100px; font-family:monospace;">' + cmd + '</textarea>' +
    '<p style="margin-top:20px;"><strong>What this does:</strong></p>' +
    '<ul>' +
    '<li>Downloads data <strong>directly from Google Drive</strong> (no size limits!)</li>' +
    '<li>Matches donor clusters to Campaign Deputy PersonIDs</li>' +
    '<li>Aggregates donation history and calculates metrics</li>' +
    '<li>Creates CD_To_Upload sheet for import</li>' +
    '<li>Automatically cleans up temporary files</li>' +
    '</ul>' +
    '<p style="margin-top:10px; color:#666; font-size:12px;">Note: CSV files are temporarily shared with "anyone with link" for download, then deleted.</p>' +
    '</div>'
  ).setWidth(600).setHeight(480);
  
  SpreadsheetApp.getUi().showModalDialog(html, 'Campaign Deputy Matching');
}
