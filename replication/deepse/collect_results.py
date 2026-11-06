#!/usr/bin/env python3
"""Collect Deep-SE test predictions into per-project CSVs + an MAE table (stdlib only).

For each project it reads
  Deep-SE/classification/<exp>/log/ar/pred_<saving>.txt   actual,predicted at the best-validation epoch (patched in)
  Deep-SE/classification/<exp>/log/<saving>.txt           Deep-SE's own per-epoch log (t_mae at best epoch)
  Deep-SE/data/files/<KEY>_deep-se_3sets.txt              which CSV rows are test rows
  datasets/Choet_Dataset/<KEY>_deep-se.csv                to recover the issue keys
"""
import argparse
import csv
import os
import statistics
import sys

csv.field_size_limit(sys.maxsize)

PROJECTS = [  # (GPT2SP name, Tawosi key)
    ('appceleratorstudio', 'TISTUD'), ('aptanastudio', 'APSTUD'), ('bamboo', 'BAM'), ('clover', 'CLOV'),
    ('datamanagement', 'DM'), ('duracloud', 'DURACLOUD'), ('jirasoftware', 'JRESERVER'), ('mesos', 'MESOS'),
    ('moodle', 'MDL'), ('mule', 'MULE'), ('mulestudio', 'MULESTUDIO'), ('springxd', 'XD'),
    ('titanium', 'TIMOB'), ('usergrid', 'USERGRID'),
]
SAVING = '{key}_deep-se_lstm_highway_dim50_reginphid_prefinetune_lm_poolmean.txt'


def own_reported(log_path):
    """Deep-SE's own log: test MAE/MdAE on the row of the best-validation epoch."""
    rows, best = {}, None
    with open(log_path) as f:
        next(f)
        for line in f:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 4:
                continue
            epoch = int(parts[0].split()[0])
            t = parts[2].split()
            rows[epoch] = (float(t[0]), float(t[1]))
            if 'Best result at epoch' in parts[3]:
                best = int(parts[3].split()[-1])
    return best, len(rows), rows[best]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--deepse', required=True)
    ap.add_argument('--exp', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'preds'), exist_ok=True)

    table = []
    for name, key in PROJECTS:
        saving = SAVING.format(key=key)
        pred_path = os.path.join(a.deepse, 'classification', a.exp, 'log', 'ar', 'pred_' + saving)
        if not os.path.isfile(pred_path):
            print('missing', key, pred_path)
            continue
        with open(pred_path) as f:
            preds = [(float(r['actual']), float(r['predicted'])) for r in csv.DictReader(f)]

        with open(os.path.join(a.data, key + '_deep-se.csv'), newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        with open(os.path.join(a.deepse, 'data', 'files', key + '_deep-se_3sets.txt')) as f:
            next(f)
            flags = [line.split() for line in f]
        assert len(flags) == len(rows), (key, len(flags), len(rows))
        test_rows = [rows[i] for i, fl in enumerate(flags) if fl[2] == '1']
        assert len(test_rows) == len(preds), (key, len(test_rows), len(preds))
        for r, (act, _) in zip(test_rows, preds):
            assert abs(float(r['storypoint']) - act) < 1e-6, (key, r['issuekey'], r['storypoint'], act)

        with open(os.path.join(a.out, 'preds', name + '_preds.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['issuekey', 'actual', 'predicted'])
            for r, (act, pr) in zip(test_rows, preds):
                w.writerow([r['issuekey'], repr(act), repr(pr)])

        ae = [abs(act - pr) for act, pr in preds]
        mae, mdae = statistics.mean(ae), statistics.median(ae)
        best, n_epochs, (own_mae, own_mdae) = own_reported(
            os.path.join(a.deepse, 'classification', a.exp, 'log', saving))
        table.append(dict(project=name, key=key, n_test=len(preds), mae='%.4f' % mae, mdae='%.4f' % mdae,
                          deepse_log_mae='%.4f' % own_mae, deepse_log_mdae='%.4f' % own_mdae,
                          best_epoch=best, n_epochs=n_epochs))

    cols = ['project', 'key', 'n_test', 'mae', 'mdae']
    with open(os.path.join(a.out, 'deepse_within.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(table)
    extra = cols + ['deepse_log_mae', 'deepse_log_mdae', 'best_epoch', 'n_epochs']
    with open(os.path.join(a.out, 'deepse_within_details.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=extra)
        w.writeheader()
        w.writerows(table)
    for t in table:
        print('{project:20s} {key:11s} n={n_test:5d} mae={mae} mdae={mdae} (log {deepse_log_mae}, best ep {best_epoch}/{n_epochs})'.format(**t))
    if table:
        print('median MAE over %d projects: %.4f' % (len(table), statistics.median(float(t['mae']) for t in table)))


if __name__ == '__main__':
    main()
