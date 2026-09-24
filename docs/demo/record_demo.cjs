// Record one continuous browser viewport while running two real investigations.
// Run with NODE_PATH pointing at an installation of playwright-core.
const fs = require('node:fs')
const path = require('node:path')
const { chromium } = require('playwright-core')

const out = process.env.SUMORA_VIDEO_OUT || '/tmp/sumora-video'
const repo = 'https://github.com/devanshranjan10/sumora-hhgoa-2026'
const started = Date.now()
const cues = []

function cue(name, speech) {
  const atMs = Date.now() - started
  cues.push({ name, atMs, speech })
  console.log(`${(atMs / 1000).toFixed(1)}s ${name}`)
}

async function holdUntil(page, seconds) {
  const remaining = seconds * 1000 - (Date.now() - started)
  if (remaining > 0) await page.waitForTimeout(remaining)
}

async function gitPage(page, suffix) {
  const response = await page.goto(`${repo}${suffix}`, { waitUntil: 'domcontentloaded', timeout: 30000 })
  if (!response || response.status() !== 200) throw new Error(`GitHub page failed: ${suffix}`)
  await page.waitForTimeout(1100)
}

async function main() {
  fs.mkdirSync(out, { recursive: true })
  const browser = await chromium.launch({
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: true,
  })
  const context = await browser.newContext({
    viewport: { width: 1600, height: 900 },
    deviceScaleFactor: 1,
    recordVideo: { dir: out, size: { width: 1600, height: 900 } },
  })
  const page = await context.newPage()
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  const video = page.video()

  try {
    const response = await page.goto('http://34.66.141.236/', { waitUntil: 'domcontentloaded', timeout: 30000 })
    if (!response || response.status() !== 200) throw new Error('Live UI did not load')
    await page.getByText('Graph + MCP + model online').waitFor({ timeout: 30000 })
    await page.getByRole('heading', { name: 'Loading case' }).waitFor({ state: 'hidden', timeout: 30000 })
    cue('intro', 'This is Sumora, our live fraud investigator. Twenty cases are the scored submission; seven more are demo imports. TigerGraph Community Edition holds five hundred and ninety thousand seven hundred and forty-seven transactions and five thousand five hundred and sixty-five closed investigations. The graph, MCP server, and decision model run together on a CPU virtual machine.')
    await holdUntil(page, 24)

    const sidebar = page.getByRole('complementary', { name: 'Case navigation' })
    await sidebar.getByRole('button', { name: 'HHG-013 legitimate' }).click()
    await page.getByRole('heading', { name: 'Loading case' }).waitFor({ state: 'hidden', timeout: 15000 })
    await page.getByRole('button', { name: 'Graph', exact: true }).click()
    await page.getByRole('img', { name: 'TigerGraph neighborhood for HHG-013' }).waitFor({ timeout: 20000 })
    cue('graph', 'Here is the live TigerGraph neighborhood for case thirteen. A flagged transaction connects to its card, device, and earlier investigations. I can select a node and inspect attributes returned by the database. A shared link is evidence for review, never an automatic fraud label.')
    await page.getByRole('button', { name: /^Flagged transaction/ }).first().click()
    await page.waitForTimeout(2500)
    const cardNode = page.getByRole('button', { name: /^Card / }).first()
    if (await cardNode.count()) await cardNode.click()
    await page.locator('.graph-inspector').scrollIntoViewIfNeeded()
    await holdUntil(page, 55)

    await page.getByRole('button', { name: 'Overview', exact: true }).click()
    cue('case13_start', 'I am now investigating case thirteen live, not replaying a saved trace. The LangGraph agent gathers graph and policy evidence, then calls installed GSQL pattern scorers through TigerGraph MCP. Watch the actual steps arrive.')
    await page.getByRole('button', { name: 'Investigate live' }).click()
    await page.getByText('LIVE / COMPLETE', { exact: true }).waitFor({ timeout: 120000 })
    await page.locator('.mcp-evidence details[open]').scrollIntoViewIfNeeded()
    cue('case13_result', 'The MCP receipt shows the installed query, its parameters, and a returned row. The calibrated model estimates nine percent fraud probability. The agent recommends monitoring, then closing this case as no fraud, with the approval route recorded.')
    await holdUntil(page, 105)

    await sidebar.getByRole('button', { name: 'HHG-011 uncertain' }).click()
    await page.getByRole('heading', { name: 'Loading case' }).waitFor({ state: 'hidden', timeout: 15000 })
    cue('case11_start', 'Case eleven is deliberately ambiguous. I am running it live to show how the agent chooses between acting now and requesting more evidence. It does not treat the supplied bank risk score as the answer.')
    await page.getByRole('button', { name: 'Investigate live' }).click()
    await page.getByText('LIVE / COMPLETE', { exact: true }).waitFor({ timeout: 120000 })
    await page.locator('.mcp-evidence details[open]').scrollIntoViewIfNeeded()
    cue('case11_result', 'The estimate is thirty-eight percent. The evidence gate finds step-up authentication worth requesting before a fraud decision. Here is another real TigerGraph MCP response. There is no customer reply, so the agent records the request as pending instead of fabricating evidence or forcing a verdict.')
    await page.waitForTimeout(6500)
    await page.getByRole('button', { name: 'Evidence', exact: true }).click()
    await page.getByText('Response status: No response supplied; request pending').scrollIntoViewIfNeeded()
    await page.waitForTimeout(5500)
    await page.getByRole('button', { name: 'Overview', exact: true }).click()
    await holdUntil(page, 160)

    await sidebar.getByRole('button', { name: /^Watchlist/ }).click()
    cue('more_data', 'The work goes beyond the twenty scored files. A time-aware graph scan has one hundred and eighty-six unimported leads. Analysts can add a case for an existing transaction or upload entirely new transaction rows. Our sixth lead type is a cross-account shared-device ring; two thousand label shuffles gave a permutation p-value of zero point zero zero zero five. These are leads, not hidden outcome labels.')
    await page.waitForTimeout(5500)
    await sidebar.getByRole('button', { name: /^Add data/ }).click()
    await page.getByRole('button', { name: 'New transaction row' }).click()
    await page.waitForTimeout(5500)
    await sidebar.getByRole('button', { name: 'HHG-011 uncertain' }).click()
    await page.getByRole('button', { name: /New typology/ }).click()
    await holdUntil(page, 200)

    await gitPage(page, '/blob/main/docs/architecture.md')
    await page.getByRole('heading', { name: 'Runtime architecture' }).waitFor({ timeout: 15000 })
    cue('architecture', 'The browser calls a Python API, never the database directly. LangGraph owns the investigation loop and its conditional gate: gather evidence or propose a policy action. The MCP client calls TigerGraph installed queries, and the trace preserves returned rows. A calibrated gradient-boosting model supplies one probability. The gate compares expected action cost with the value of more evidence; policy guards, approval routes, and graph write-back are executable code.')
    await page.waitForTimeout(7500)
    await gitPage(page, '/blob/main/agent/runner.py#L331-L364')
    await page.waitForTimeout(8500)
    await gitPage(page, '/blob/main/agent/mcp_graph.py#L26-L37')
    await page.waitForTimeout(7000)
    await gitPage(page, '/blob/main/agent/gate.py#L178-L202')
    await holdUntil(page, 251)

    await gitPage(page, '/tree/main/cases')
    cue('results', 'The public repository contains exactly twenty answer files at the root under cases. All twenty passed the answer checks and were written back to TigerGraph. Our predictions are ten fraud, six legitimate, and four uncertain; those are not known outcomes. On two hundred and ninety-five historical October cases where both labels exist, the model reached A U C zero point nine two seven and class-balanced Brier zero point one zero nine. Hidden challenge accuracy is unknown.')
    await page.waitForTimeout(9000)
    await gitPage(page, '/blob/main/docs/evaluation.md')
    await page.getByRole('heading', { name: 'Replacement decision model' }).scrollIntoViewIfNeeded()
    await holdUntil(page, 286)

    if (errors.length) console.warn('Page errors:', errors)
  } finally {
    fs.writeFileSync(path.join(out, 'cues.json'), JSON.stringify(cues, null, 2))
    await context.close()
    const videoPath = await video.path()
    fs.copyFileSync(videoPath, path.join(out, 'continuous.webm'))
    await browser.close()
    console.log('Recorded video:', path.join(out, 'continuous.webm'))
    console.log('Duration clock:', ((Date.now() - started) / 1000).toFixed(1), 'seconds')
  }
}

main().catch(error => {
  console.error(error)
  process.exitCode = 1
})
