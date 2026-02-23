/** Action handlers split for safer per-area changes */

function dl_evolveBlockingRules_() {
  const ss = SpreadsheetApp.getActive();
  const ui = SpreadsheetApp.getUi();

  // Check if BlockingRules sheet exists
  let sheet = ss.getSheetByName('BlockingRules');
  if (!sheet || sheet.getLastRow() < 2) {
    ui.alert(
      'No Blocking Rules Found',
      'The BlockingRules sheet is empty. The sheet will be populated with 50 default rules on next open. ' +
      'Please close and reopen the spreadsheet to initialize the sheet.',
      ui.ButtonSet.OK
    );
    return;
  }

  // Get web app URL
  const webAppUrl = dl_getExecUrlFromOptions_();
  if (!webAppUrl) {
    ui.alert(
      'Deploy this bound Apps Script as a Web App (Deploy → Manage deployments), then re-open the sheet.\nOptional override: set Options!I2 to a specific /exec URL.'
    );
    return;
  }

  // Export KREF and FEC sheets to Drive (needed for CSV endpoints)
  const kref = dl_exportSheetAsCsv_(DL_CFG.krefSheet);
  const fec  = dl_exportSheetAsCsv_(DL_CFG.fecSheet);
  if (!kref.file || !fec.file) {
    ui.alert(
      'Missing input sheets or no rows. Confirm sheets exist: ' + DL_CFG.krefSheet + ' and ' + DL_CFG.fecSheet
    );
    return;
  }

  // Generate token for data access
  const token = dl_makeToken_();
  const until = Date.now() + DL_CFG.tokenMinutes * 60 * 1000;

  const props = PropertiesService.getDocumentProperties();
  props.setProperty('dl_token', token);
  props.setProperty('dl_token_until', String(until));
  props.setProperty('dl_kref_fileId', kref.file.getId());
  props.setProperty('dl_fec_fileId', fec.file.getId());

  // Build URLs
  const modelUrl = webAppUrl + '?model=1';
  const krefUrl = webAppUrl + '?csv=kref&token=' + encodeURIComponent(token);
  const fecUrl = webAppUrl + '?csv=fec&token=' + encodeURIComponent(token);
  const rulesUrl = webAppUrl + '?csv=blocking_rules&token=' + encodeURIComponent(token);

  // Build command that downloads script first, then runs it (so stdin is available for input())
  const cmd =
    "curl -sSL '" + webAppUrl + "?evolve_runner=1' -o /tmp/evolve_rules.py && " +
    "python3 /tmp/evolve_rules.py " +
    "--model-url '" + modelUrl + "' " +
    "--kref-url '" + krefUrl + "' " +
    "--fec-url '" + fecUrl + "' " +
    "--rules-url '" + rulesUrl + "'";

  // Show command dialog
  const html = HtmlService.createHtmlOutput(
    '<div style="font-family:system-ui,Arial;padding:12px;max-width:720px">' +
      '<h3 style="margin:0 0 12px 0">🧬 Evolve Blocking Rules</h3>' +
      '<div style="margin:6px 0">Token expires in about ' + DL_CFG.tokenMinutes + ' minutes</div>' +
      '<div style="margin:6px 0">Copy this command into your Mac Terminal:</div>' +
      '<textarea id="cmd" style="width:100%;height:160px;font-family:monospace;font-size:11px" readonly>' +
        dl_htmlEscape_(cmd) +
      '</textarea>' +
      '<div style="margin-top:8px">' +
        '<button onclick="navigator.clipboard.writeText(document.getElementById(\'cmd\').value)">Copy to clipboard</button>' +
      '</div>' +
      '<div style="margin-top:12px;padding:10px;background:#fff3cd;border-left:3px solid #ffc107;font-size:12px">' +
        '<strong>How it works:</strong> Evolution will continuously improve your 50 blocking rules ' +
        'organized into 5 cohorts (10 rules each).' +
      '</div>' +
      '<div style="margin-top:8px;padding:10px;background:#ffebee;border-left:3px solid #f44336;font-size:12px">' +
        '<strong>⚠️ To stop evolution:</strong> Open a NEW Terminal window and run:<br>' +
        '<code style="display:block;margin-top:4px;padding:4px;background:#fff">touch /tmp/blocking_rules_stop</code>' +
        '<div style="margin-top:4px;font-size:11px">The evolution will finish the current round and save results.</div>' +
      '</div>' +
      '<div style="margin-top:8px;padding:10px;background:#e8f5e9;border-left:3px solid #4caf50;font-size:12px">' +
        '<strong>✓ Results auto-saved to:</strong> <code>~/Downloads/blocking_rules_evolved.csv</code><br>' +
        'Open the file in your Downloads folder, copy all rows (including header), ' +
        'and paste into BlockingRules sheet (A1) to replace existing rules.' +
      '</div>' +
      '<div style="margin-top:8px;color:#666;font-size:11px">' +
        'Evolution scores rules by finding matches within a 100,000 comparison budget. ' +
        'Cohort 1 optimizes for pure performance, cohorts 2-5 maintain diversity.' +
      '</div>' +
    '</div>'
  ).setWidth(740).setHeight(480);

  ui.showModalDialog(html, '🧬 Evolve Blocking Rules');
}
