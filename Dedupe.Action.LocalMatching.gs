/** Action handlers split for safer per-area changes */

function dl_prepareLocalJobAndShowCommand_() {
  dl_resetProgress_();
  dl_setProgress_('Preparing job', 0, 1, 'Creating Drive bundle');

  const webAppUrl = dl_getExecUrlFromOptions_();
  if (!webAppUrl) {
    SpreadsheetApp.getUi().alert(
      'Deploy this bound Apps Script as a Web App (Deploy → Manage deployments), then re-open the sheet.\nOptional override: set Options!I2 to a specific /exec URL.'
    );
    return;
  }

  const kref = dl_exportSheetAsCsv_(DL_CFG.krefSheet);
  const fec  = dl_exportSheetAsCsv_(DL_CFG.fecSheet);
  if (!kref.file || !fec.file) {
    SpreadsheetApp.getUi().alert(
      'Missing input sheets or no rows. Confirm sheets exist: ' + DL_CFG.krefSheet + ' and ' + DL_CFG.fecSheet
    );
    return;
  }

  const token = dl_makeToken_();
  const until = Date.now() + DL_CFG.tokenMinutes * 60 * 1000;

  // Read match threshold from Options!J2
  const ss = SpreadsheetApp.getActive();
  const optionsSheet = ss.getSheetByName('Options');
  const threshold = optionsSheet ? Number(optionsSheet.getRange('J2').getValue() || 0.7) : 0.7;

  const job = {
    jobId: 'job_' + Utilities.getUuid().replace(/-/g, '').slice(0, 12),
    cfg: {
      sampleTrainingPairs: DL_CFG.sampleTrainingPairs,
      uncertainBatchSize: DL_CFG.uncertainBatchSize,
      uncertaintyBand: DL_CFG.uncertaintyBand,
      maxPairsPerBlock: DL_CFG.maxPairsPerBlock,
      maxTotalPairs: DL_CFG.maxTotalPairs,
      predictThreshold: threshold
    },
    krefUrl: webAppUrl + '?csv=kref&token=' + encodeURIComponent(token),
    fecUrl:  webAppUrl + '?csv=fec&token=' + encodeURIComponent(token),
    modelUrl: webAppUrl + '?model=1',
    resultUrl: webAppUrl + '?result=1&token=' + encodeURIComponent(token)
  };

  const props = PropertiesService.getDocumentProperties();
  props.setProperty('dl_token', token);
  props.setProperty('dl_token_until', String(until));
  props.setProperty('dl_job_json', JSON.stringify(job));
  props.setProperty('dl_kref_fileId', kref.file.getId());
  props.setProperty('dl_fec_fileId',  fec.file.getId());
  props.setProperty('dl_staging_folderId', kref.folder.getId());

  const runnerVersion = String(Date.now());

  const cmd =
    "echo '=== Web app health endpoint ===' && " +
    "curl -sSL '" + webAppUrl + "?health=1&v=" + runnerVersion + "' -o /tmp/donor_health.json && " +
    "cat /tmp/donor_health.json && echo && " +
    "grep -q '\"ok\":true' /tmp/donor_health.json || { " +
      "echo 'ERROR: web app health endpoint did not return Apps Script JSON.'; " +
      "echo 'If you see HTML, this /exec URL is not your active Web App deployment.'; " +
      "exit 1; " +
    "} && " +
    "grep -q 'DM_LOCAL_RUNNER_20260222' /tmp/donor_health.json || { " +
      "echo 'ERROR: health fingerprint mismatch (old deployment or wrong URL).'; " +
      "exit 1; " +
    "} && " +
    "curl -sSL '" + webAppUrl + "?runner=1&v=" + runnerVersion + "' -o /tmp/donor_runner.py && " +
    "echo '=== Runner header (first 5 lines) ===' && head -n 5 /tmp/donor_runner.py && " +
    "echo '=== Runner fingerprint lines ===' && (grep -n 'RUNNER_FINGERPRINT' /tmp/donor_runner.py || echo 'No fingerprint found in fetched runner') && " +
    "grep -q 'RUNNER_FINGERPRINT: DM_LOCAL_RUNNER_20260222' /tmp/donor_runner.py || { " +
      "echo 'ERROR: fetched runner is not the expected deployment (fingerprint missing/mismatched).'; " +
      "echo 'Fix: redeploy web app and ensure Options!I2 points to that /exec URL.'; " +
      "exit 1; " +
    "} && " +
    "python3 -m py_compile /tmp/donor_runner.py || { " +
      "echo '=== RUNNER PYTHON SYNTAX CHECK FAILED ==='; " +
      "echo 'Showing lines 2918-2928 for diagnosis:'; " +
      "nl -ba /tmp/donor_runner.py | sed -n '2918,2928p'; " +
      "echo 'First suspicious print(\" occurrence:'; " +
      "grep -n 'print(\"' /tmp/donor_runner.py | head -n 1; " +
      "echo 'Tip: if fingerprint is missing above, you are hitting an old deployment URL/version.'; " +
      "exit 1; " +
    "} && " +
    "echo '=== Job bundle preflight ===' && " +
    "curl -sSL '" + webAppUrl + "?job=1&token=" + token + "' -o /tmp/donor_job.json && " +
    "grep -q '\"jobId\"' /tmp/donor_job.json || { " +
      "echo 'ERROR: job bundle endpoint did not return expected JSON (token invalid/expired).'; " +
      "echo 'Job endpoint response:'; cat /tmp/donor_job.json; " +
      "exit 1; " +
    "} && " +
    "python3 /tmp/donor_runner.py --bundle '" + webAppUrl + "?job=1&token=" + token + "' " +
    "--result '" + webAppUrl + "?result=1&token=" + token + "'";

  dl_setProgress_('Waiting', 0, 0, 'Copy the command into Terminal');

  const html = HtmlService.createHtmlOutput(
    '<div style="font-family:system-ui,Arial;padding:12px;max-width:720px">' +
      '<div style="margin:6px 0">Token expires in about ' + DL_CFG.tokenMinutes + ' minutes</div>' +
      '<div style="margin:6px 0">Copy this command into your Mac Terminal:</div>' +
      '<textarea id="cmd" style="width:100%;height:140px" readonly>' +
        dl_htmlEscape_(cmd) +
      '</textarea>' +
      '<div style="margin-top:8px">' +
        '<button onclick="navigator.clipboard.writeText(document.getElementById(\'cmd\').value)">Copy to clipboard</button>' +
      '</div>' +
      '<div style="margin-top:8px;color:#666;font-size:12px">Keep this sheet open. Progress will appear in the sidebar.</div>' +
    '</div>'
  ).setWidth(740).setHeight(280);
  SpreadsheetApp.getUi().showModalDialog(html, 'Run locally');
}
function dl_showProgressSidebar() {
  const html = HtmlService.createHtmlOutput(
    '<div style="font-family:system-ui,Arial;padding:12px;max-width:360px">' +
      '<h3 style="margin:0 0 8px 0;font-weight:600">Local run progress</h3>' +
      '<div id="p">No updates yet.</div>' +
      '<script>' +
      '  function fmt(p){ if(!p) return "No updates yet."; ' +
      '    return "Phase: " + (p.phase||"") + "<br>Done: " + (p.done||0) + " of " + (p.total||0) + "<br>Note: " + (p.note||"") + "<br><small>"+new Date(p.ts).toLocaleString()+"</small>"; }' +
      '  function poll(){ google.script.run.withSuccessHandler(function(s){ ' +
      '      try{var p=s?JSON.parse(s):null; document.getElementById("p").innerHTML = fmt(p);}catch(e){document.getElementById("p").textContent="Parse error";}' +
      '    }).dl_getProgressPayload_(); }' +
      '  setInterval(poll, 1500); poll();' +
      '</script>' +
    '</div>'
  ).setTitle('Local run progress');
  SpreadsheetApp.getUi().showSidebar(html);
}
