package tinydecide

import "math"

// Example is what the model returned for a message, stored under the option a
// person said was right (for noul questions, list 1 is "true" and list 0 is
// "false").
type Example struct {
	V []float64 // answer.Qvec
	Z []float64 // answer.Z0
}

// ExampleFromAnswer builds an Example from an Answer.
func ExampleFromAnswer(a *Answer) Example {
	v := make([]float64, len(a.Qvec))
	for i, x := range a.Qvec {
		v[i] = float64(x)
	}
	return Example{V: v, Z: append([]float64(nil), a.Z0...)}
}

// Lambdas are the trust weights tried by LambdaFor.
var Lambdas = [5]float64{0.0, 0.25, 0.5, 1.0, 2.0}

// MeanVec returns the mean of the example vectors over dh dimensions.
func MeanVec(list []*Example, dh int) []float64 {
	m := make([]float64, dh)
	n := float64(len(list))
	for _, e := range list {
		for i := 0; i < dh && i < len(e.V); i++ {
			m[i] += e.V[i] / n
		}
	}
	return m
}

func cosC(a, b, c []float64) float64 {
	var d, na, nb float64
	for i := 0; i < len(a); i++ {
		x := a[i] - c[i]
		y := b[i] - c[i]
		d += x * y
		na += x * x
		nb += y * y
	}
	n := math.Sqrt(na * nb)
	if n == 0 || math.IsNaN(n) {
		n = 1e-12
	}
	return d / n
}

func termFor(v []float64, lists [][]*Example, c, beta []float64) []float64 {
	out := make([]float64, len(lists))
	for i, l := range lists {
		if len(l) == 0 {
			out[i] = 0
		} else {
			out[i] = beta[bucket(len(l))] * cosC(v, MeanVec(l, len(v)), c)
		}
	}
	return out
}

// LambdaFor chooses how much to trust a question's corrections via a
// leave-one-out log-likelihood over the examples.
func LambdaFor(kind QType, lists [][]Example, c, beta []float64) float64 {
	type idx struct {
		k, j int
		e    *Example
	}
	var all []idx
	for k := range lists {
		for j := range lists[k] {
			all = append(all, idx{k, j, &lists[k][j]})
		}
	}
	if len(all) < 2 {
		return 0.25
	}
	best, bestLL := 0.0, math.Inf(-1)
	for _, lam := range Lambdas {
		var ll float64
		for _, a := range all {
			rest := make([][]*Example, len(lists))
			for kk := range lists {
				for jj := range lists[kk] {
					if kk == a.k && jj == a.j {
						continue
					}
					rest[kk] = append(rest[kk], &lists[kk][jj])
				}
			}
			t := termFor(a.e.V, rest, c, beta)
			var z []float64
			if kind == Noul {
				z1 := 0.0
				if len(a.e.Z) > 1 {
					z1 = a.e.Z[1]
				}
				z = []float64{lam * sliceAt(t, 0), z1 + lam*sliceAt(t, 1)}
			} else {
				z = make([]float64, len(a.e.Z))
				for i, x := range a.e.Z {
					z[i] = x + lam*sliceAtNaN(t, i)
				}
			}
			mmax := math.Inf(-1)
			for _, x := range z {
				if x > mmax {
					mmax = x
				}
			}
			var sum float64
			for _, x := range z {
				sum += math.Exp(x - mmax)
			}
			lse := mmax + math.Log(sum)
			ll += sliceAtNaN(z, a.k) - lse
		}
		if ll > bestLL+1e-9 {
			best = lam
			bestLL = ll
		}
	}
	return best
}

func sliceAt(s []float64, i int) float64 {
	if i < len(s) {
		return s[i]
	}
	return 0
}

func sliceAtNaN(s []float64, i int) float64 {
	if i < len(s) {
		return s[i]
	}
	return math.NaN()
}

// MakeProtos turns stored examples into the Protos argument of AnswerWith.
// lists is one list of examples per option (2 for noul); beta is TinyDecide.Beta;
// center is an optional running mean of qvec over every message asked with this
// question. Returns nil when there are no examples (span questions take none).
func MakeProtos(kind QType, lists [][]Example, beta []float64, center []float64) *Protos {
	if kind == Span {
		return nil
	}
	dh := -1
	for _, l := range lists {
		if len(l) > 0 {
			dh = len(l[0].V)
			break
		}
	}
	if dh < 0 {
		return nil
	}
	var c []float64
	if center != nil {
		c = append([]float64(nil), center...)
	} else {
		var flat []*Example
		for i := range lists {
			for j := range lists[i] {
				flat = append(flat, &lists[i][j])
			}
		}
		c = MeanVec(flat, dh)
	}
	vec := make([]float32, len(lists)*dh)
	cnt := make([]int, len(lists))
	for k := range lists {
		cnt[k] = len(lists[k])
		if len(lists[k]) > 0 {
			ptrs := make([]*Example, len(lists[k]))
			for j := range lists[k] {
				ptrs[j] = &lists[k][j]
			}
			m := MeanVec(ptrs, dh)
			for i, x := range m {
				vec[k*dh+i] = float32(x)
			}
		}
	}
	lam := LambdaFor(kind, lists, c, beta)
	cf := make([]float32, len(c))
	for i, x := range c {
		cf[i] = float32(x)
	}
	return &Protos{Vec: vec, Cnt: cnt, Center: cf, Lam: &lam}
}
