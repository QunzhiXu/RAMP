"""Meaningful checks for loss gradients, identity boundaries and sklearn parity."""
import sys
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPRegressor
from rank_head import pairs, ranking_loss_gradient, SourceRankScore, train, predict
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from trainability.scoring import regression_score


def main():
    rng = np.random.RandomState(731)
    measured = np.array([1., 1., 4., 2., 7., 7., 0.])
    units = np.array(['a', 'a', 'a', 'b', 'b', 'b', 'c'])
    ps = pairs(measured, units)
    assert np.all(units[ps[0]] == units[ps[1]])
    assert np.all(measured[ps[0]] != measured[ps[1]])
    assert len(ps[0]) == 4 and np.isclose(ps[3].sum(), 1)
    prediction = rng.normal(size=len(measured))
    loss, grad = ranking_loss_gradient(prediction, ps)
    eps = 1e-6; numeric = []
    for i in range(len(measured)):
        plus = prediction.copy(); minus = prediction.copy()
        plus[i] += eps; minus[i] -= eps
        numeric.append((ranking_loss_gradient(plus, ps)[0] - ranking_loss_gradient(minus, ps)[0]) / (2*eps))
    np.testing.assert_allclose(grad, numeric, rtol=1e-6, atol=1e-8)
    assert ranking_loss_gradient(np.ones(7), pairs(np.ones(7), units))[0] == 0
    assert abs(grad.sum()) < 1e-12
    assert ranking_loss_gradient(prediction - .001*grad, ps)[0] < loss

    x = rng.normal(size=(500, 8)); y = rng.uniform(size=500)
    vx = rng.normal(size=(512, 8)); truth = rng.randint(0, 25, 512).astype(float)
    vu = np.repeat(['u1','u2','u3','u4'], 128)
    vp = np.repeat(['p1','p1','p2','p3'], 128)
    score = SourceRankScore(truth, vu, vp)
    for pred in [rng.normal(size=512), np.zeros(512)]:
        assert abs(score(pred)-regression_score(truth,pred,vu,vp)['effective_spearman']) < 1e-12
    c = dict(width=32,learning_rate=.001,alpha=.001,batch_size=128,max_epochs=40,patience=15)
    params, info = train(x,y,y,np.repeat(['f1','f2'],250),vx,score,c,20260920,0)
    m = MLPRegressor(hidden_layer_sizes=(32,),random_state=20260920,batch_size=128,
        early_stopping=False,learning_rate_init=.001,alpha=.001,solver='adam')
    best=-np.inf; stale=0
    for epoch in range(40):
        m.partial_fit(x,y); s=score(m.predict(vx))
        if s>best:
            best=s; expected=m.predict(vx).copy(); selected=epoch+1; stale=0
        else: stale+=1
        if stale>=15: break
    np.testing.assert_allclose(predict(params,vx),expected,rtol=1e-10,atol=1e-10)
    assert info['selected_epoch']==selected and abs(info['source_validation_score']-best)<1e-12
    print('PASS: finite-difference ranking gradient; within-unit and tie controls; source scoring; sklearn baseline parity')


if __name__ == '__main__': main()
