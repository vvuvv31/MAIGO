"""Reconstruct full-family paths without dropping descendant deposits.

Geometry only. Descendant births must lie on a recorded parent segment; no
invented connector or closest-track alias is permitted. No transport fitting.
"""
import numpy as np

class OrderedPaths:
    def __init__(self,rows):
        self.rows=rows;self.indices={};self.prefixes={};self.visiting=set()
        for i,r in enumerate(rows):self.indices.setdefault(tuple(r[:3].astype(int)),[]).append(i)
        self.maximum_birth_residual_mm=0.

    def track(self,key):
        if key not in self.indices:raise ValueError('Missing ancestor track')
        ids=self.indices[key];r=self.rows[ids]
        if len(np.unique(r[:,5]))!=len(r) or np.any(np.diff(r[:,5])<=0):raise ValueError('Unordered or duplicate track steps')
        if len(r)>1 and np.max(np.linalg.norm(r[1:,10:13]-r[:-1,13:16],axis=1))>1e-6:raise ValueError('Track discontinuity')
        if np.linalg.norm(r[0,6:9]-r[0,10:13])>1e-6:raise ValueError('Birth to first step discontinuity')
        return ids,r

    def prefix(self,key):
        if key in self.prefixes:return self.prefixes[key]
        if key in self.visiting:raise ValueError('Cyclic ancestry')
        self.visiting.add(key)
        ids,r=self.track(key);parent=(key[0],key[1],int(r[0,3]))
        if parent not in self.indices:raise ValueError('Missing parent')
        pi,p=self.track(parent)
        if p[0,3]==0:
            if r[0,4]!=11:raise ValueError('Non-electron primary daughter')
            out=np.empty((0,3))
        else:
            origin=r[0,6:9];delta=p[:,13:16]-p[:,10:13]
            length2=np.sum(delta**2,axis=1)
            t=np.divide(np.sum((origin-p[:,10:13])*delta,axis=1),length2,
                        out=np.zeros(len(p)),where=length2>0)
            t=np.clip(t,0,1);closest=p[:,10:13]+t[:,None]*delta
            residual=np.linalg.norm(closest-origin,axis=1);best=int(np.argmin(residual))
            if residual[best]>1e-6:raise ValueError(f'Birth not on parent polyline: {residual[best]:.8g} mm')
            self.maximum_birth_residual_mm=max(self.maximum_birth_residual_mm,float(residual[best]))
            # Consecutive shared endpoints are equivalent. A spatial self-
            # intersection at a distinct earlier time is ambiguous: reject.
            hits=np.flatnonzero(residual<1e-8)
            if len(hits)>1 and np.linalg.norm(delta[hits.min()+1:hits.max()],axis=1).sum()>1e-6:
                raise ValueError('Ambiguous parent birth position')
            out=np.concatenate((self.prefix(parent),delta[:best],(t[best]*delta[best])[None,:]))
        self.visiting.remove(key);self.prefixes[key]=out
        return out

    def path(self,row_index,fraction):
        if not 0<=fraction<=1:raise ValueError('Deposit fraction')
        row=self.rows[row_index];key=tuple(row[:3].astype(int));ids,r=self.track(key)
        j=ids.index(row_index)
        delta=r[:,13:16]-r[:,10:13]
        result=np.concatenate((self.prefix(key),delta[:j],(fraction*delta[j])[None,:]))
        return result
