// All three instantiations of the templated library in one binary, plus the historical
// (non-template) interface.
// Build: c++ -std=c++11 -I cpp/gco tests/cpp_all_types.cpp -o /tmp/t
#include <stdio.h>
#include "GCoptimization.cpp"
#include "LinkedBlockList.cpp"

static int failures = 0;

static void check(const char* what, double got, double want)
{
	bool ok = (got == want);
	printf("%-28s %12g %s\n", what, got, ok ? "ok" : "FAILED");
	if (!ok)
		++failures;
}

// A 4-site chain whose optimal labeling is (0,1,0,0) with E=4.
template <typename Term>
static double chain(const char* name)
{
	Term D[12] = {0,3,5, 9,0,9, 2,3,0, 0,3,5};
	Term V[9]  = {0,1,2, 1,0,1, 2,1,0};
	GCoptimizationGeneralGraphT<Term> g(4, 3);
	g.setDataCost(D);
	g.setSmoothCost(V);
	g.setNeighbors(0, 1, (Term)1);
	g.setNeighbors(1, 2, (Term)1);
	g.setNeighbors(2, 3, (Term)2);
	double e = (double)g.expansion();
	int l[4];
	g.whatLabel(0, 4, l);
	printf("%s labeling: %d %d %d %d\n", name, l[0], l[1], l[2], l[3]);
	if (l[0] != 0 || l[1] != 1 || l[2] != 0 || l[3] != 0) {
		printf("%s labeling FAILED\n", name);
		++failures;
	}
	return e;
}

// energy.h's own documented example, plus a triple term (Energy::add_term3).
static void energy_h_example()
{
	typedef Energy<int,int,long long> E;
	E e(3, 2);
	E::Var x = e.add_variable(), y = e.add_variable(), z = e.add_variable();
	e.add_term1(x, 0, 1);
	e.add_term1(y, 0, -2);
	e.add_term1(z, 3, 0);
	e.add_term2(x, y, 0, 0, 0, -4);
	e.add_term2(y, z, 0, 5, 5, 0);
	check("energy.h example", (double)e.minimize(), -5);
	printf("energy.h solution: %d %d %d\n", e.get_var(x), e.get_var(y), e.get_var(z));
	if (e.get_var(x) != 1 || e.get_var(y) != 1 || e.get_var(z) != 1) {
		printf("energy.h solution FAILED\n");
		++failures;
	}

	E t(3, 8);
	E::Var a = t.add_variable(), b = t.add_variable(), c = t.add_variable();
	t.add_term1(a, 0, 1);
	t.add_term1(b, 0, 1);
	t.add_term1(c, 0, 1);
	t.add_term3(a, b, c, 0,0,0,0,0,0,0,-4);   // a + b + c - 4*a*b*c
	check("energy.h add_term3", (double)t.minimize(), -1);
}

int main()
{
	check("int32 chain energy", chain<int>("int32"), 4);
	check("int64 chain energy", chain<long long>("int64"), 4);
	check("float64 chain energy", chain<double>("float64"), 4);

	// Each type has its own term ceiling.
	check("int32 max term", (double)GCoptimizationT<int>::maxEnergyTerm(), 1e7);
	check("int64 max term", (double)GCoptimizationT<long long>::maxEnergyTerm(), 1e12);
	check("float64 max term", GCoptimizationT<double>::maxEnergyTerm(), 1e7);

	// Non-template names still resolve, and default to int32 terms / int64 totals.
	static_assert(std::is_same<GCoptimization, GCoptimizationT<int> >::value, "legacy alias");
	static_assert(std::is_same<GCoptimization::EnergyType, long long>::value, "legacy totals");
	static_assert(std::is_same<GCoptimization::EnergyTermType, int>::value, "legacy terms");
	static_assert(std::is_same<GCoptimizationGridGraph, GCoptimizationGridGraphT<int> >::value, "legacy grid");
	static_assert(std::is_same<GCoptimizationGeneralGraph, GCoptimizationGeneralGraphT<int> >::value, "legacy general");

	// A grid problem through the historical interface.
	GCoptimizationGridGraph grid(3, 3, 2);
	GCoptimization::EnergyTermType D[18], V[4] = {0,1, 1,0};
	for (int s = 0; s < 9; ++s) {
		D[s*2 + 0] = (s < 4) ? 0 : 4;
		D[s*2 + 1] = (s < 4) ? 4 : 0;
	}
	grid.setDataCost(D);
	grid.setSmoothCost(V);
	check("legacy grid energy", (double)grid.expansion(), 4);  // 4 cut edges

	energy_h_example();

	printf(failures ? "\nFAILED\n" : "\nPASSED\n");
	return failures ? 1 : 0;
}
