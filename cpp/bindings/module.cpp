// CPython extension for GCoptimization. All three energy-term types (int32, int64
// and float64) are compiled in; a GCO object picks one at construction time.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <type_traits>
#include <vector>

#if PY_VERSION_HEX < 0x030D0000
#error Requires Python 3.13 or later.
#endif

extern "C" bool gco_py_interrupt();
#define GCO_INTERRUPT_FN gco_py_interrupt

#include "GCoptimization.cpp"
#include "LinkedBlockList.cpp"

// The historical, non-template names must keep working for C++ users.
static_assert(std::is_same<GCoptimization, GCoptimizationT<int> >::value, "legacy alias changed");
static_assert(std::is_same<GCoptimization::EnergyType, long long>::value, "legacy total type changed");
static_assert(std::is_same<GCoptimizationGridGraph, GCoptimizationGridGraphT<int> >::value, "legacy alias changed");
static_assert(std::is_same<GCoptimizationGeneralGraph, GCoptimizationGeneralGraphT<int> >::value, "legacy alias changed");

extern "C" bool gco_py_interrupt()
{
	PyGILState_STATE g = PyGILState_Ensure();
	bool pending = PyErr_CheckSignals() != 0;
	PyGILState_Release(g);
	return pending;
}

// Releases the GIL, then takes the per-object lock; reverse on scope exit.
struct Unlocked {
	Unlocked(std::mutex* m): m_save(PyEval_SaveThread()), m_mu(m) { m_mu->lock(); }
	~Unlocked() { m_mu->unlock(); PyEval_RestoreThread(m_save); }
private:
	PyThreadState* m_save;
	std::mutex* m_mu;
};

/////////////////////////////////////////////////////////////////////
// Type-erased view of GCoptimizationT<int> / GCoptimizationT<double>

struct EnergyVal {
	long long i;
	double d;
	bool isf;
	PyObject* toPy() const { return isf ? PyFloat_FromDouble(d) : PyLong_FromLongLong(i); }
};

struct Backend {
	virtual ~Backend() { }
	virtual void setDataCost(void* costs) = 0;
	virtual void setDataCostSparse(int label, const int* sites, const void* costs, Py_ssize_t n) = 0;
	virtual void setSmoothCost(void* costs) = 0;
	virtual void setSmoothCostVH(void* costs, void* vWeights, void* hWeights) = 0;
	virtual void setNeighbors(const int* s1, const int* s2, const void* w, Py_ssize_t n) = 0;
	virtual void setLabelCost(double cost) = 0;
	virtual void setLabelCostArray(const void* costs) = 0;
	virtual void setLabelSubsetCost(const int* labels, int n, double cost) = 0;
	virtual void setLabeling(const int* labeling, int n) = 0;
	virtual void getLabeling(int start, int count, int* out) = 0;
	virtual void setLabelOrderRandom(bool isRandom) = 0;
	virtual void setLabelOrder(const int* order, int n) = 0;
	virtual void setVerbosity(int level) = 0;
	virtual EnergyVal expansion(int maxCycles) = 0;
	virtual bool alphaExpansion(int alpha) = 0;
	virtual EnergyVal swap(int maxCycles) = 0;
	virtual void alphaBetaSwap(int alpha, int beta) = 0;
	virtual EnergyVal computeEnergy() = 0;
	virtual EnergyVal dataEnergy() = 0;
	virtual EnergyVal smoothEnergy() = 0;
	virtual EnergyVal labelEnergy() = 0;
};

template <typename Term>
struct BackendT: Backend {
	typedef GCoptimizationT<Term> G;

	BackendT(int numSites, int numLabels, int width, int height)
	: m_g(width ? (G*)new GCoptimizationGridGraphT<Term>(width, height, numLabels)
	            : (G*)new GCoptimizationGeneralGraphT<Term>(numSites, numLabels))
	{ }
	~BackendT() { delete m_g; }

	void setDataCost(void* costs) { m_g->setDataCost((Term*)costs); }
	void setDataCostSparse(int label, const int* sites, const void* costs, Py_ssize_t n)
	{
		std::vector<typename G::SparseDataCost> sparse((size_t)n);
		for (Py_ssize_t i = 0; i < n; ++i) {
			sparse[(size_t)i].site = sites[i];
			sparse[(size_t)i].cost = ((const Term*)costs)[i];
		}
		m_g->setDataCost(label, n ? &sparse[0] : NULL, (int)n);
	}
	void setSmoothCost(void* costs) { m_g->setSmoothCost((Term*)costs); }
	void setSmoothCostVH(void* costs, void* vWeights, void* hWeights)
	{
		((GCoptimizationGridGraphT<Term>*)m_g)->setSmoothCostVH((Term*)costs, (Term*)vWeights, (Term*)hWeights);
	}
	void setNeighbors(const int* s1, const int* s2, const void* w, Py_ssize_t n)
	{
		GCoptimizationGeneralGraphT<Term>* g = (GCoptimizationGeneralGraphT<Term>*)m_g;
		for (Py_ssize_t i = 0; i < n; ++i)
			g->setNeighbors(s1[i], s2[i], w ? ((const Term*)w)[i] : (Term)1);
	}
	void setLabelCost(double cost) { m_g->setLabelCost((Term)cost); }
	void setLabelCostArray(const void* costs) { m_g->setLabelCost((Term*)costs); }
	void setLabelSubsetCost(const int* labels, int n, double cost) { m_g->setLabelSubsetCost((int*)labels, n, (Term)cost); }
	void setLabeling(const int* labeling, int n)
	{
		for (int i = 0; i < n; ++i)
			m_g->setLabel(i, labeling[i]);
	}
	void getLabeling(int start, int count, int* out) { m_g->whatLabel(start, count, out); }
	void setLabelOrderRandom(bool isRandom) { m_g->setLabelOrder(isRandom); }
	void setLabelOrder(const int* order, int n) { m_g->setLabelOrder(order, n); }
	void setVerbosity(int level) { m_g->setVerbosity(level); }
	EnergyVal expansion(int maxCycles) { return wrap(m_g->expansion(maxCycles)); }
	bool alphaExpansion(int alpha) { return m_g->alpha_expansion(alpha); }
	EnergyVal swap(int maxCycles) { return wrap(m_g->swap(maxCycles)); }
	void alphaBetaSwap(int alpha, int beta) { m_g->alpha_beta_swap(alpha, beta); }
	EnergyVal computeEnergy() { return wrap(m_g->compute_energy()); }
	EnergyVal dataEnergy() { return wrap(m_g->giveDataEnergy()); }
	EnergyVal smoothEnergy() { return wrap(m_g->giveSmoothEnergy()); }
	EnergyVal labelEnergy() { return wrap(m_g->giveLabelEnergy()); }

private:
	static EnergyVal wrap(typename G::EnergyType e)
	{
		EnergyVal v;
		v.isf = std::is_same<Term, double>::value;
		v.i = (long long)e;
		v.d = (double)e;
		return v;
	}
	G* m_g;
};

/////////////////////////////////////////////////////////////////////
// Python object

// dtype codes, shared with bkgco/_core.py
enum { GCO_INT32 = 0, GCO_INT64 = 1, GCO_FLOAT64 = 2 };

struct GCOObj {
	PyObject_HEAD
	Backend* b;
	std::mutex* mu;
	int dtype_code;
	int is_float;
	int grid;
	int ns;
	int nl;
	int width;
	int height;
	Py_buffer dc, sc;
};

static const char* term_fmt(GCOObj* self)
{
	return self->dtype_code == GCO_INT32 ? "il" : (self->dtype_code == GCO_INT64 ? "lq" : "d");
}
static Py_ssize_t term_size(GCOObj* self) { return self->dtype_code == GCO_INT32 ? 4 : 8; }
static const char kIdxFmt[] = "il";

static PyObject* raise_gc(const char* msg)
{
	if (!PyErr_Occurred())
		PyErr_SetString(PyExc_RuntimeError, msg);
	return NULL;
}

#define GCO_CATCH \
	catch (GCException& x) { return raise_gc(x.message); } \
	catch (std::bad_alloc&) { return PyErr_NoMemory(); } \
	catch (...) { return raise_gc("unknown error inside GCoptimization"); }

static int term_from_py(GCOObj* self, PyObject* o, double* out)
{
	double v = PyFloat_AsDouble(o);
	if (v == -1.0 && PyErr_Occurred())
		return -1;
	if (!self->is_float) {
		// Terms are exchanged as double; every term the library accepts is exactly
		// representable (the int64 ceiling is well under 2**53).
		double lo = self->dtype_code == GCO_INT32 ? -2147483648.0 : -9007199254740992.0;
		double hi = self->dtype_code == GCO_INT32 ? 2147483647.0 : 9007199254740992.0;
		if (v != (double)(long long)v || v < lo || v > hi) {
			PyErr_Format(PyExc_ValueError, "cost must be an integer within %s range",
			             self->dtype_code == GCO_INT32 ? "int32" : "int64");
			return -1;
		}
	}
	*out = v;
	return 0;
}

static int fmt_ok(const char* f, const char* accept)
{
	const unsigned short one = 1;
	const char native = (*(const unsigned char*)&one) ? '<' : '>';
	if (!f)
		return 0;
	if (*f == '@' || *f == '=' || *f == native)
		++f;
	return f[0] && !f[1] && strchr(accept, f[0]) != NULL;
}

// count < 0 accepts any length; returns element count or -1 on error.
static Py_ssize_t getbuf(PyObject* o, Py_buffer* b, int writable, const char* accept,
                         Py_ssize_t itemsize, Py_ssize_t count, const char* what)
{
	int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT | (writable ? PyBUF_WRITABLE : 0);
	if (PyObject_GetBuffer(o, b, flags) < 0)
		return -1;
	if (b->itemsize != itemsize || !fmt_ok(b->format, accept)) {
		PyBuffer_Release(b);
		PyErr_Format(PyExc_TypeError, "%s: expected a contiguous array of %zd-byte '%s' elements",
		             what, itemsize, accept);
		return -1;
	}
	Py_ssize_t n = b->len / itemsize;
	if (count >= 0 && n != count) {
		PyBuffer_Release(b);
		PyErr_Format(PyExc_ValueError, "%s: expected %zd elements, got %zd", what, count, n);
		return -1;
	}
	return n;
}

static int check_open(GCOObj* self)
{
	if (!self->b) {
		PyErr_SetString(PyExc_RuntimeError, "GCO object has been destroyed");
		return 0;
	}
	return 1;
}

/////////////////////////////////////////////////////////////////////
// methods

static PyObject* m_destroy(GCOObj* self, PyObject* Py_UNUSED(a))
{
	if (self->b) {
		Unlocked u(self->mu);  // wait for any in-flight operation
		delete self->b;
		self->b = NULL;
	}
	PyBuffer_Release(&self->dc);
	PyBuffer_Release(&self->sc);
	Py_RETURN_NONE;
}

static PyObject* m_set_data_cost(GCOObj* self, PyObject* arg)
{
	if (!check_open(self))
		return NULL;
	Py_buffer b;
	if (getbuf(arg, &b, 0, term_fmt(self), term_size(self),
	           (Py_ssize_t)self->ns * self->nl, "data cost") < 0)
		return NULL;
	try {
		Unlocked u(self->mu);
		self->b->setDataCost(b.buf);
	} catch (GCException& x) {
		PyBuffer_Release(&b);
		return raise_gc(x.message);
	} catch (...) {
		PyBuffer_Release(&b);
		return raise_gc("unknown error inside GCoptimization");
	}
	PyBuffer_Release(&self->dc);
	self->dc = b;
	Py_RETURN_NONE;
}

static PyObject* m_set_data_cost_sparse(GCOObj* self, PyObject* args)
{
	int label;
	PyObject* sites_o;
	PyObject* costs_o;
	if (!PyArg_ParseTuple(args, "iOO", &label, &sites_o, &costs_o))
		return NULL;
	if (!check_open(self))
		return NULL;
	if (label < 0 || label >= self->nl) {
		PyErr_SetString(PyExc_ValueError, "label out of range");
		return NULL;
	}
	Py_buffer sb, cb;
	Py_ssize_t n = getbuf(sites_o, &sb, 0, kIdxFmt, 4, -1, "sparse sites");
	if (n < 0)
		return NULL;
	if (getbuf(costs_o, &cb, 0, term_fmt(self), term_size(self), n, "sparse costs") < 0) {
		PyBuffer_Release(&sb);
		return NULL;
	}
	PyObject* res = NULL;
	try {
		Unlocked u(self->mu);
		self->b->setDataCostSparse(label, (const int*)sb.buf, cb.buf, n);
		res = Py_None;
	} catch (GCException& x) {
		raise_gc(x.message);
	} catch (std::bad_alloc&) {
		PyErr_NoMemory();
	} catch (...) {
		raise_gc("unknown error inside GCoptimization");
	}
	PyBuffer_Release(&sb);
	PyBuffer_Release(&cb);
	return res ? Py_NewRef(res) : NULL;
}

static PyObject* m_set_smooth_cost(GCOObj* self, PyObject* arg)
{
	if (!check_open(self))
		return NULL;
	Py_buffer b;
	if (getbuf(arg, &b, 0, term_fmt(self), term_size(self),
	           (Py_ssize_t)self->nl * self->nl, "smooth cost") < 0)
		return NULL;
	try {
		Unlocked u(self->mu);
		self->b->setSmoothCost(b.buf);
	} catch (GCException& x) {
		PyBuffer_Release(&b);
		return raise_gc(x.message);
	} catch (...) {
		PyBuffer_Release(&b);
		return raise_gc("unknown error inside GCoptimization");
	}
	PyBuffer_Release(&self->sc);
	self->sc = b;
	Py_RETURN_NONE;
}

static PyObject* m_set_smooth_cost_vh(GCOObj* self, PyObject* args)
{
	PyObject* v_o;
	PyObject* vc_o;
	PyObject* hc_o;
	if (!PyArg_ParseTuple(args, "OOO", &v_o, &vc_o, &hc_o))
		return NULL;
	if (!check_open(self))
		return NULL;
	if (!self->grid) {
		PyErr_SetString(PyExc_RuntimeError, "set_smooth_cost_vh requires a grid graph");
		return NULL;
	}
	Py_buffer vb, vcb, hcb;
	if (getbuf(v_o, &vb, 0, term_fmt(self), term_size(self),
	           (Py_ssize_t)self->nl * self->nl, "smooth cost") < 0)
		return NULL;
	if (getbuf(vc_o, &vcb, 0, term_fmt(self), term_size(self), self->ns, "vertical weights") < 0) {
		PyBuffer_Release(&vb);
		return NULL;
	}
	if (getbuf(hc_o, &hcb, 0, term_fmt(self), term_size(self), self->ns, "horizontal weights") < 0) {
		PyBuffer_Release(&vb);
		PyBuffer_Release(&vcb);
		return NULL;
	}
	int ok = 0;
	try {
		Unlocked u(self->mu);
		self->b->setSmoothCostVH(vb.buf, vcb.buf, hcb.buf);
		ok = 1;
	} catch (GCException& x) {
		raise_gc(x.message);
	} catch (std::bad_alloc&) {
		PyErr_NoMemory();
	} catch (...) {
		raise_gc("unknown error inside GCoptimization");
	}
	PyBuffer_Release(&vcb);  // weights are copied internally
	PyBuffer_Release(&hcb);
	if (!ok) {
		PyBuffer_Release(&vb);
		return NULL;
	}
	PyBuffer_Release(&self->sc);
	self->sc = vb;
	Py_RETURN_NONE;
}

static PyObject* m_set_neighbors(GCOObj* self, PyObject* args)
{
	PyObject* s1_o;
	PyObject* s2_o;
	PyObject* w_o;
	if (!PyArg_ParseTuple(args, "OOO", &s1_o, &s2_o, &w_o))
		return NULL;
	if (!check_open(self))
		return NULL;
	if (self->grid) {
		PyErr_SetString(PyExc_RuntimeError, "set_neighbors requires a general graph");
		return NULL;
	}
	Py_buffer b1, b2, wb;
	Py_ssize_t n = getbuf(s1_o, &b1, 0, kIdxFmt, 4, -1, "edge sites");
	if (n < 0)
		return NULL;
	if (getbuf(s2_o, &b2, 0, kIdxFmt, 4, n, "edge sites") < 0) {
		PyBuffer_Release(&b1);
		return NULL;
	}
	int have_w = (w_o != Py_None);
	if (have_w && getbuf(w_o, &wb, 0, term_fmt(self), term_size(self), n, "edge weights") < 0) {
		PyBuffer_Release(&b1);
		PyBuffer_Release(&b2);
		return NULL;
	}
	const int* s1 = (const int*)b1.buf;
	const int* s2 = (const int*)b2.buf;
	PyObject* res = NULL;
	for (Py_ssize_t i = 0; i < n; ++i) {
		if (s1[i] < 0 || s1[i] >= self->ns || s2[i] < 0 || s2[i] >= self->ns) {
			PyErr_Format(PyExc_ValueError, "edge %zd: site index out of range", i);
			goto done;
		}
		if (s1[i] == s2[i]) {
			PyErr_Format(PyExc_ValueError, "edge %zd: a site cannot neighbor itself", i);
			goto done;
		}
	}
	try {
		Unlocked u(self->mu);
		self->b->setNeighbors(s1, s2, have_w ? wb.buf : NULL, n);
		res = Py_None;
	} catch (GCException& x) {
		raise_gc(x.message);
	} catch (std::bad_alloc&) {
		PyErr_NoMemory();
	} catch (...) {
		raise_gc("unknown error inside GCoptimization");
	}
done:
	PyBuffer_Release(&b1);
	PyBuffer_Release(&b2);
	if (have_w)
		PyBuffer_Release(&wb);
	return res ? Py_NewRef(res) : NULL;
}

static PyObject* m_set_label_cost(GCOObj* self, PyObject* arg)
{
	if (!check_open(self))
		return NULL;
	double c;
	if (term_from_py(self, arg, &c) < 0)
		return NULL;
	try {
		Unlocked u(self->mu);
		self->b->setLabelCost(c);
	}
	GCO_CATCH
	Py_RETURN_NONE;
}

static PyObject* m_set_label_cost_array(GCOObj* self, PyObject* arg)
{
	if (!check_open(self))
		return NULL;
	Py_buffer b;
	if (getbuf(arg, &b, 0, term_fmt(self), term_size(self), self->nl, "label cost") < 0)
		return NULL;
	PyObject* res = NULL;
	try {
		Unlocked u(self->mu);
		self->b->setLabelCostArray(b.buf);  // copied internally
		res = Py_None;
	} catch (GCException& x) {
		raise_gc(x.message);
	} catch (...) {
		raise_gc("unknown error inside GCoptimization");
	}
	PyBuffer_Release(&b);
	return res ? Py_NewRef(res) : NULL;
}

static PyObject* m_set_label_subset_cost(GCOObj* self, PyObject* args)
{
	PyObject* labels_o;
	PyObject* cost_o;
	if (!PyArg_ParseTuple(args, "OO", &labels_o, &cost_o))
		return NULL;
	if (!check_open(self))
		return NULL;
	double c;
	if (term_from_py(self, cost_o, &c) < 0)
		return NULL;
	Py_buffer b;
	Py_ssize_t n = getbuf(labels_o, &b, 0, kIdxFmt, 4, -1, "label subset");
	if (n < 0)
		return NULL;
	PyObject* res = NULL;
	if (n < 1) {
		PyErr_SetString(PyExc_ValueError, "label subset must be non-empty");
	} else {
		try {
			Unlocked u(self->mu);
			self->b->setLabelSubsetCost((const int*)b.buf, (int)n, c);  // copied internally
			res = Py_None;
		} catch (GCException& x) {
			raise_gc(x.message);
		} catch (...) {
			raise_gc("unknown error inside GCoptimization");
		}
	}
	PyBuffer_Release(&b);
	return res ? Py_NewRef(res) : NULL;
}

static PyObject* m_set_labeling(GCOObj* self, PyObject* arg)
{
	if (!check_open(self))
		return NULL;
	Py_buffer b;
	if (getbuf(arg, &b, 0, kIdxFmt, 4, self->ns, "labeling") < 0)
		return NULL;
	const int* l = (const int*)b.buf;
	for (int i = 0; i < self->ns; ++i) {
		if (l[i] < 0 || l[i] >= self->nl) {
			PyBuffer_Release(&b);
			PyErr_Format(PyExc_ValueError, "labeling[%d] out of range 0..%d", i, self->nl - 1);
			return NULL;
		}
	}
	{
		Unlocked u(self->mu);
		self->b->setLabeling(l, self->ns);
	}
	PyBuffer_Release(&b);
	Py_RETURN_NONE;
}

static PyObject* m_get_labeling(GCOObj* self, PyObject* args)
{
	PyObject* out_o;
	int start = 0;
	if (!PyArg_ParseTuple(args, "O|i", &out_o, &start))
		return NULL;
	if (!check_open(self))
		return NULL;
	Py_buffer b;
	Py_ssize_t n = getbuf(out_o, &b, 1, kIdxFmt, 4, -1, "labeling output");
	if (n < 0)
		return NULL;
	if (start < 0 || start + n > self->ns) {
		PyBuffer_Release(&b);
		PyErr_SetString(PyExc_ValueError, "requested site range is out of bounds");
		return NULL;
	}
	{
		Unlocked u(self->mu);
		self->b->getLabeling(start, (int)n, (int*)b.buf);
	}
	PyBuffer_Release(&b);
	Py_RETURN_NONE;
}

static PyObject* m_set_label_order_random(GCOObj* self, PyObject* arg)
{
	if (!check_open(self))
		return NULL;
	int is_random = PyObject_IsTrue(arg);
	if (is_random < 0)
		return NULL;
	Unlocked u(self->mu);
	self->b->setLabelOrderRandom(is_random != 0);
	Py_RETURN_NONE;
}

static PyObject* m_set_label_order(GCOObj* self, PyObject* arg)
{
	if (!check_open(self))
		return NULL;
	Py_buffer b;
	Py_ssize_t n = getbuf(arg, &b, 0, kIdxFmt, 4, -1, "label order");
	if (n < 0)
		return NULL;
	PyObject* res = NULL;
	if (n > self->nl) {
		PyErr_SetString(PyExc_ValueError, "label order has more entries than labels");
	} else {
		try {
			Unlocked u(self->mu);
			self->b->setLabelOrder((const int*)b.buf, (int)n);
			res = Py_None;
		} catch (GCException& x) {
			raise_gc(x.message);
		} catch (...) {
			raise_gc("unknown error inside GCoptimization");
		}
	}
	PyBuffer_Release(&b);
	return res ? Py_NewRef(res) : NULL;
}

static PyObject* m_set_verbosity(GCOObj* self, PyObject* arg)
{
	if (!check_open(self))
		return NULL;
	long level = PyLong_AsLong(arg);
	if (level == -1 && PyErr_Occurred())
		return NULL;
	if (level < 0 || level > 2) {
		PyErr_SetString(PyExc_ValueError, "verbosity must be 0, 1 or 2");
		return NULL;
	}
	self->b->setVerbosity((int)level);
	Py_RETURN_NONE;
}

static PyObject* m_expansion(GCOObj* self, PyObject* args)
{
	int max_cycles = -1;
	if (!PyArg_ParseTuple(args, "|i", &max_cycles))
		return NULL;
	if (!check_open(self))
		return NULL;
	EnergyVal e;
	try {
		Unlocked u(self->mu);
		e = self->b->expansion(max_cycles);
	}
	GCO_CATCH
	return e.toPy();
}

static PyObject* m_swap(GCOObj* self, PyObject* args)
{
	int max_cycles = -1;
	if (!PyArg_ParseTuple(args, "|i", &max_cycles))
		return NULL;
	if (!check_open(self))
		return NULL;
	EnergyVal e;
	try {
		Unlocked u(self->mu);
		e = self->b->swap(max_cycles);
	}
	GCO_CATCH
	return e.toPy();
}

static PyObject* m_alpha_expansion(GCOObj* self, PyObject* arg)
{
	if (!check_open(self))
		return NULL;
	long alpha = PyLong_AsLong(arg);
	if (alpha == -1 && PyErr_Occurred())
		return NULL;
	if (alpha < 0 || alpha >= self->nl) {
		PyErr_SetString(PyExc_ValueError, "label out of range");
		return NULL;
	}
	bool changed;
	try {
		Unlocked u(self->mu);
		changed = self->b->alphaExpansion((int)alpha);
	}
	GCO_CATCH
	return PyBool_FromLong(changed);
}

static PyObject* m_alpha_beta_swap(GCOObj* self, PyObject* args)
{
	int alpha, beta;
	if (!PyArg_ParseTuple(args, "ii", &alpha, &beta))
		return NULL;
	if (!check_open(self))
		return NULL;
	if (alpha < 0 || alpha >= self->nl || beta < 0 || beta >= self->nl) {
		PyErr_SetString(PyExc_ValueError, "label out of range");
		return NULL;
	}
	if (alpha == beta) {
		PyErr_SetString(PyExc_ValueError, "alpha and beta must differ");
		return NULL;
	}
	try {
		Unlocked u(self->mu);
		self->b->alphaBetaSwap(alpha, beta);
	}
	GCO_CATCH
	Py_RETURN_NONE;
}

#define GCO_ENERGY_GETTER(name, call) \
	static PyObject* name(GCOObj* self, PyObject* Py_UNUSED(a)) \
	{ \
		if (!check_open(self)) \
			return NULL; \
		EnergyVal e; \
		try { \
			Unlocked u(self->mu); \
			e = self->b->call(); \
		} \
		GCO_CATCH \
		return e.toPy(); \
	}

GCO_ENERGY_GETTER(m_compute_energy, computeEnergy)
GCO_ENERGY_GETTER(m_data_energy, dataEnergy)
GCO_ENERGY_GETTER(m_smooth_energy, smoothEnergy)
GCO_ENERGY_GETTER(m_label_energy, labelEnergy)

static PyMethodDef gco_methods[] = {
	{"destroy", (PyCFunction)m_destroy, METH_NOARGS, NULL},
	{"set_data_cost", (PyCFunction)m_set_data_cost, METH_O, NULL},
	{"set_data_cost_sparse", (PyCFunction)m_set_data_cost_sparse, METH_VARARGS, NULL},
	{"set_smooth_cost", (PyCFunction)m_set_smooth_cost, METH_O, NULL},
	{"set_smooth_cost_vh", (PyCFunction)m_set_smooth_cost_vh, METH_VARARGS, NULL},
	{"set_neighbors", (PyCFunction)m_set_neighbors, METH_VARARGS, NULL},
	{"set_label_cost", (PyCFunction)m_set_label_cost, METH_O, NULL},
	{"set_label_cost_array", (PyCFunction)m_set_label_cost_array, METH_O, NULL},
	{"set_label_subset_cost", (PyCFunction)m_set_label_subset_cost, METH_VARARGS, NULL},
	{"set_labeling", (PyCFunction)m_set_labeling, METH_O, NULL},
	{"get_labeling", (PyCFunction)m_get_labeling, METH_VARARGS, NULL},
	{"set_label_order_random", (PyCFunction)m_set_label_order_random, METH_O, NULL},
	{"set_label_order", (PyCFunction)m_set_label_order, METH_O, NULL},
	{"set_verbosity", (PyCFunction)m_set_verbosity, METH_O, NULL},
	{"expansion", (PyCFunction)m_expansion, METH_VARARGS, NULL},
	{"swap", (PyCFunction)m_swap, METH_VARARGS, NULL},
	{"alpha_expansion", (PyCFunction)m_alpha_expansion, METH_O, NULL},
	{"alpha_beta_swap", (PyCFunction)m_alpha_beta_swap, METH_VARARGS, NULL},
	{"compute_energy", (PyCFunction)m_compute_energy, METH_NOARGS, NULL},
	{"data_energy", (PyCFunction)m_data_energy, METH_NOARGS, NULL},
	{"smooth_energy", (PyCFunction)m_smooth_energy, METH_NOARGS, NULL},
	{"label_energy", (PyCFunction)m_label_energy, METH_NOARGS, NULL},
	{NULL, NULL, 0, NULL}
};

static PyMemberDef gco_members[] = {
	{"num_sites", Py_T_INT, offsetof(GCOObj, ns), Py_READONLY, NULL},
	{"num_labels", Py_T_INT, offsetof(GCOObj, nl), Py_READONLY, NULL},
	{"width", Py_T_INT, offsetof(GCOObj, width), Py_READONLY, NULL},
	{"height", Py_T_INT, offsetof(GCOObj, height), Py_READONLY, NULL},
	{"is_grid", Py_T_INT, offsetof(GCOObj, grid), Py_READONLY, NULL},
	{"is_float64", Py_T_INT, offsetof(GCOObj, is_float), Py_READONLY, NULL},
	{"dtype_code", Py_T_INT, offsetof(GCOObj, dtype_code), Py_READONLY, NULL},
	{NULL, 0, 0, 0, NULL}
};

static int gco_init(PyObject* op, PyObject* args, PyObject* kwds)
{
	static const char* kwlist[] = {"num_sites", "num_labels", "dtype_code", "width", "height", NULL};
	GCOObj* self = (GCOObj*)op;
	int ns = 0, nl = 0, code = GCO_INT32, width = 0, height = 0;
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "ii|iii", (char**)kwlist,
	                                 &ns, &nl, &code, &width, &height))
		return -1;
	if (code < GCO_INT32 || code > GCO_FLOAT64) {
		PyErr_SetString(PyExc_ValueError, "dtype_code must be 0 (int32), 1 (int64) or 2 (float64)");
		return -1;
	}
	if (self->b) {
		PyErr_SetString(PyExc_RuntimeError, "GCO object is already initialized");
		return -1;
	}
	if (ns < 1 || nl < 2) {
		PyErr_SetString(PyExc_ValueError, "num_sites must be >= 1 and num_labels >= 2");
		return -1;
	}
	if (width || height) {
		if (width < 2 || height < 2 || (double)width * height != (double)ns) {
			PyErr_SetString(PyExc_ValueError, "grid requires width >= 2, height >= 2 and num_sites == width*height");
			return -1;
		}
	}
	try {
		self->mu = new std::mutex();
		switch (code) {
		case GCO_INT64:   self->b = new BackendT<long long>(ns, nl, width, height); break;
		case GCO_FLOAT64: self->b = new BackendT<double>(ns, nl, width, height); break;
		default:          self->b = new BackendT<int>(ns, nl, width, height); break;
		}
	} catch (GCException& x) {
		raise_gc(x.message);
		return -1;
	} catch (std::bad_alloc&) {
		PyErr_NoMemory();
		return -1;
	}
	self->ns = ns;
	self->nl = nl;
	self->width = width;
	self->height = height;
	self->grid = (width || height) ? 1 : 0;
	self->dtype_code = code;
	self->is_float = (code == GCO_FLOAT64) ? 1 : 0;
	return 0;
}

static void gco_dealloc(PyObject* op)
{
	GCOObj* self = (GCOObj*)op;
	PyTypeObject* tp = Py_TYPE(op);
	delete self->b;
	self->b = NULL;
	PyBuffer_Release(&self->dc);
	PyBuffer_Release(&self->sc);
	delete self->mu;
	self->mu = NULL;
	tp->tp_free(op);
	Py_DECREF(tp);
}

static PyType_Slot gco_slots[] = {
	{Py_tp_new, (void*)PyType_GenericNew},
	{Py_tp_init, (void*)gco_init},
	{Py_tp_dealloc, (void*)gco_dealloc},
	{Py_tp_methods, (void*)gco_methods},
	{Py_tp_members, (void*)gco_members},
	{Py_tp_doc, (void*)"Low-level handle to a GCoptimization instance (int32 or float64 terms)."},
	{0, NULL}
};

static PyType_Spec gco_spec = {
	"bkgco._bkgco.GCO",
	sizeof(GCOObj),
	0,
	Py_TPFLAGS_DEFAULT,
	gco_slots
};

/////////////////////////////////////////////////////////////////////
// BKEnergy: Kolmogorov-Zabih Energy<> (energy.h) over binary variables,
// fed in batches. Variables are created once, up front; add_term3's
// auxiliary variables are appended after them and stay private.

struct BKError {
	BKError(const std::string& m): msg(m) { }
	std::string msg;
};

static void bk_error(const char* msg) { throw GCException(msg); }

static std::string bk_msg(Py_ssize_t row, const char* what)
{
	char buf[256];
	snprintf(buf, sizeof(buf), "row %zd: %s", row, what);
	return std::string(buf);
}

struct EnergyBackend {
	virtual ~EnergyBackend() { }
	virtual void addTerm1(const void* vals, Py_ssize_t n) = 0;
	virtual void addTerm2(const int* idx, const void* vals, Py_ssize_t n) = 0;
	virtual void addTerm3(const int* idx, const void* vals, Py_ssize_t n) = 0;
	virtual void addConstant(double c) = 0;
	virtual EnergyVal minimize() = 0;
	virtual void getSolution(unsigned char* out, int n) = 0;
};

template <typename Term>
struct EnergyBackendT: EnergyBackend {
	typedef Energy<Term, Term, typename GCO_Traits<Term>::TotalType> E;

	EnergyBackendT(int numVars): m_e(new E(numVars, 16, bk_error))
	{
		m_e->add_variable(numVars);
	}
	~EnergyBackendT() { delete m_e; }

	void addTerm1(const void* vals, Py_ssize_t n)
	{
		const Term* v = (const Term*)vals;
		for (Py_ssize_t i = 0; i < n; ++i) {
			check(v[2*i], i);
			check(v[2*i+1], i);
			m_e->add_term1((int)i, v[2*i], v[2*i+1]);
		}
	}

	void addTerm2(const int* idx, const void* vals, Py_ssize_t n)
	{
		const Term* v = (const Term*)vals;
		for (Py_ssize_t i = 0; i < n; ++i) {
			const Term* t = v + 4*i;
			for (int k = 0; k < 4; ++k)
				check(t[k], i);
			// E00 + E11 <= E01 + E10
			if (t[0] + t[3] > t[1] + t[2])
				throw BKError(bk_msg(i, "pairwise term is not regular (needs E00 + E11 <= E01 + E10)"));
			m_e->add_term2(idx[2*i], idx[2*i+1], t[0], t[1], t[2], t[3]);
		}
	}

	void addTerm3(const int* idx, const void* vals, Py_ssize_t n)
	{
		const Term* v = (const Term*)vals;
		for (Py_ssize_t i = 0; i < n; ++i) {
			const Term* t = v + 8*i;  // E000 E001 E010 E011 E100 E101 E110 E111
			for (int k = 0; k < 8; ++k)
				check(t[k], i);
			// Regular iff every projection onto two variables is regular (6 of them).
			static const int lhs[6][2] = {{0,3},{4,7},{0,5},{2,7},{0,6},{1,7}};
			static const int rhs[6][2] = {{1,2},{5,6},{1,4},{3,6},{2,4},{3,5}};
			static const char* fixed[6] = {"x=0","x=1","y=0","y=1","z=0","z=1"};
			for (int k = 0; k < 6; ++k) {
				if (t[lhs[k][0]] + t[lhs[k][1]] > t[rhs[k][0]] + t[rhs[k][1]]) {
					char buf[128];
					snprintf(buf, sizeof(buf),
					         "triple term is not regular (projection with %s is not regular)", fixed[k]);
					throw BKError(bk_msg(i, buf));
				}
			}
			m_e->add_term3(idx[3*i], idx[3*i+1], idx[3*i+2],
			               t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
		}
	}

	void addConstant(double c) { m_e->add_constant((Term)c); }

	EnergyVal minimize()
	{
		EnergyVal v;
		v.isf = std::is_same<Term, double>::value;
		typename E::TotalValue e = m_e->minimize();
		v.i = (long long)e;
		v.d = (double)e;
		return v;
	}

	void getSolution(unsigned char* out, int n)
	{
		for (int i = 0; i < n; ++i)
			out[i] = (unsigned char)m_e->get_var(i);
	}

private:
	static void check(Term v, Py_ssize_t row)
	{
		Term m = GCO_Traits<Term>::maxTerm();
		if (v > m || v < -m)
			throw BKError(bk_msg(row, "energy term magnitude exceeds max_energy_term(dtype)"));
	}
	E* m_e;
};

struct BKObj {
	PyObject_HEAD
	EnergyBackend* e;
	std::mutex* mu;
	int dtype_code;
	int is_float;
	int nvars;
	int solved;
	EnergyVal energy;
};

static const char* bk_term_fmt(BKObj* self)
{
	return self->dtype_code == GCO_INT32 ? "il" : (self->dtype_code == GCO_INT64 ? "lq" : "d");
}
static Py_ssize_t bk_term_size(BKObj* self) { return self->dtype_code == GCO_INT32 ? 4 : 8; }

static int bk_check_open(BKObj* self)
{
	if (!self->e) {
		PyErr_SetString(PyExc_RuntimeError, "BKEnergy object has been destroyed");
		return 0;
	}
	return 1;
}

static int bk_check_editable(BKObj* self)
{
	if (!bk_check_open(self))
		return 0;
	if (self->solved) {
		PyErr_SetString(PyExc_RuntimeError, "cannot add terms after minimize()");
		return 0;
	}
	return 1;
}

#define BK_CATCH \
	catch (BKError& x) { PyErr_SetString(PyExc_ValueError, x.msg.c_str()); return NULL; } \
	catch (GCException& x) { return raise_gc(x.message); } \
	catch (std::bad_alloc&) { return PyErr_NoMemory(); } \
	catch (...) { return raise_gc("unknown error inside Energy"); }

// Reads an (n, cols) index array and checks every entry against nvars.
static Py_ssize_t bk_getidx(BKObj* self, PyObject* o, Py_buffer* b, int cols, Py_ssize_t rows)
{
	Py_ssize_t n = getbuf(o, b, 0, kIdxFmt, 4, rows < 0 ? -1 : rows * cols, "variable indices");
	if (n < 0)
		return -1;
	if (n % cols) {
		PyBuffer_Release(b);
		PyErr_Format(PyExc_ValueError, "variable indices: expected %d columns", cols);
		return -1;
	}
	const int* idx = (const int*)b->buf;
	for (Py_ssize_t i = 0; i < n / cols; ++i) {
		for (int k = 0; k < cols; ++k) {
			if (idx[i*cols + k] < 0 || idx[i*cols + k] >= self->nvars) {
				PyBuffer_Release(b);
				PyErr_Format(PyExc_ValueError, "row %zd: variable index out of range 0..%d",
				             i, self->nvars - 1);
				return -1;
			}
			for (int j = 0; j < k; ++j) {
				if (idx[i*cols + k] == idx[i*cols + j]) {
					PyBuffer_Release(b);
					PyErr_Format(PyExc_ValueError, "row %zd: a term must use distinct variables", i);
					return -1;
				}
			}
		}
	}
	return n / cols;
}

static PyObject* bk_add_term1(BKObj* self, PyObject* arg)
{
	if (!bk_check_editable(self))
		return NULL;
	Py_buffer b;
	if (getbuf(arg, &b, 0, bk_term_fmt(self), bk_term_size(self),
	           (Py_ssize_t)self->nvars * 2, "unary terms") < 0)
		return NULL;
	try {
		Unlocked u(self->mu);
		self->e->addTerm1(b.buf, self->nvars);
	} catch (BKError& x) {
		PyBuffer_Release(&b);
		PyErr_SetString(PyExc_ValueError, x.msg.c_str());
		return NULL;
	} catch (GCException& x) {
		PyBuffer_Release(&b);
		return raise_gc(x.message);
	} catch (...) {
		PyBuffer_Release(&b);
		return raise_gc("unknown error inside Energy");
	}
	PyBuffer_Release(&b);
	Py_RETURN_NONE;
}

// shared by add_term2 (arity 2, 4 values) and add_term3 (arity 3, 8 values)
static PyObject* bk_add_termN(BKObj* self, PyObject* args, int arity)
{
	PyObject* idx_o;
	PyObject* val_o;
	if (!PyArg_ParseTuple(args, "OO", &idx_o, &val_o))
		return NULL;
	if (!bk_check_editable(self))
		return NULL;
	Py_buffer ib, vb;
	Py_ssize_t rows = bk_getidx(self, idx_o, &ib, arity, -1);
	if (rows < 0)
		return NULL;
	int ncost = 1 << arity;
	if (getbuf(val_o, &vb, 0, bk_term_fmt(self), bk_term_size(self), rows * ncost, "energy terms") < 0) {
		PyBuffer_Release(&ib);
		return NULL;
	}
	PyObject* res = NULL;
	try {
		Unlocked u(self->mu);
		if (arity == 2)
			self->e->addTerm2((const int*)ib.buf, vb.buf, rows);
		else
			self->e->addTerm3((const int*)ib.buf, vb.buf, rows);
		res = Py_None;
	} catch (BKError& x) {
		PyErr_SetString(PyExc_ValueError, x.msg.c_str());
	} catch (GCException& x) {
		raise_gc(x.message);
	} catch (std::bad_alloc&) {
		PyErr_NoMemory();
	} catch (...) {
		raise_gc("unknown error inside Energy");
	}
	PyBuffer_Release(&ib);
	PyBuffer_Release(&vb);
	return res ? Py_NewRef(res) : NULL;
}

static PyObject* bk_add_term2(BKObj* self, PyObject* args) { return bk_add_termN(self, args, 2); }
static PyObject* bk_add_term3(BKObj* self, PyObject* args) { return bk_add_termN(self, args, 3); }

static PyObject* bk_add_constant(BKObj* self, PyObject* arg)
{
	if (!bk_check_editable(self))
		return NULL;
	double v = PyFloat_AsDouble(arg);
	if (v == -1.0 && PyErr_Occurred())
		return NULL;
	if (!self->is_float && v != (double)(long long)v) {
		PyErr_SetString(PyExc_ValueError, "constant must be an integer for integer dtypes");
		return NULL;
	}
	try {
		Unlocked u(self->mu);
		self->e->addConstant(v);
	}
	BK_CATCH
	Py_RETURN_NONE;
}

static PyObject* bk_minimize(BKObj* self, PyObject* Py_UNUSED(a))
{
	if (!bk_check_open(self))
		return NULL;
	if (!self->solved) {
		EnergyVal v;
		try {
			Unlocked u(self->mu);
			v = self->e->minimize();
		}
		BK_CATCH
		self->energy = v;
		self->solved = 1;
	}
	return self->energy.toPy();
}

static PyObject* bk_get_solution(BKObj* self, PyObject* arg)
{
	if (!bk_check_open(self))
		return NULL;
	if (!self->solved) {
		PyErr_SetString(PyExc_RuntimeError, "call minimize() before get_solution()");
		return NULL;
	}
	Py_buffer b;
	if (getbuf(arg, &b, 1, "B", 1, self->nvars, "solution output") < 0)
		return NULL;
	{
		Unlocked u(self->mu);
		self->e->getSolution((unsigned char*)b.buf, self->nvars);
	}
	PyBuffer_Release(&b);
	Py_RETURN_NONE;
}

static PyObject* bk_destroy(BKObj* self, PyObject* Py_UNUSED(a))
{
	if (self->e) {
		Unlocked u(self->mu);
		delete self->e;
		self->e = NULL;
	}
	Py_RETURN_NONE;
}

static PyMethodDef bk_methods[] = {
	{"add_term1", (PyCFunction)bk_add_term1, METH_O, NULL},
	{"add_term2", (PyCFunction)bk_add_term2, METH_VARARGS, NULL},
	{"add_term3", (PyCFunction)bk_add_term3, METH_VARARGS, NULL},
	{"add_constant", (PyCFunction)bk_add_constant, METH_O, NULL},
	{"minimize", (PyCFunction)bk_minimize, METH_NOARGS, NULL},
	{"get_solution", (PyCFunction)bk_get_solution, METH_O, NULL},
	{"destroy", (PyCFunction)bk_destroy, METH_NOARGS, NULL},
	{NULL, NULL, 0, NULL}
};

static PyMemberDef bk_members[] = {
	{"num_vars", Py_T_INT, offsetof(BKObj, nvars), Py_READONLY, NULL},
	{"dtype_code", Py_T_INT, offsetof(BKObj, dtype_code), Py_READONLY, NULL},
	{"is_float64", Py_T_INT, offsetof(BKObj, is_float), Py_READONLY, NULL},
	{"solved", Py_T_INT, offsetof(BKObj, solved), Py_READONLY, NULL},
	{NULL, 0, 0, 0, NULL}
};

static int bk_init(PyObject* op, PyObject* args, PyObject* kwds)
{
	static const char* kwlist[] = {"num_vars", "dtype_code", NULL};
	BKObj* self = (BKObj*)op;
	int nvars = 0, code = GCO_INT32;
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "i|i", (char**)kwlist, &nvars, &code))
		return -1;
	if (self->e) {
		PyErr_SetString(PyExc_RuntimeError, "BKEnergy object is already initialized");
		return -1;
	}
	if (nvars < 1) {
		PyErr_SetString(PyExc_ValueError, "num_vars must be >= 1");
		return -1;
	}
	if (code < GCO_INT32 || code > GCO_FLOAT64) {
		PyErr_SetString(PyExc_ValueError, "dtype_code must be 0 (int32), 1 (int64) or 2 (float64)");
		return -1;
	}
	try {
		self->mu = new std::mutex();
		switch (code) {
		case GCO_INT64:   self->e = new EnergyBackendT<long long>(nvars); break;
		case GCO_FLOAT64: self->e = new EnergyBackendT<double>(nvars); break;
		default:          self->e = new EnergyBackendT<int>(nvars); break;
		}
	} catch (GCException& x) {
		raise_gc(x.message);
		return -1;
	} catch (std::bad_alloc&) {
		PyErr_NoMemory();
		return -1;
	}
	self->nvars = nvars;
	self->dtype_code = code;
	self->is_float = (code == GCO_FLOAT64) ? 1 : 0;
	self->solved = 0;
	return 0;
}

static void bk_dealloc(PyObject* op)
{
	BKObj* self = (BKObj*)op;
	PyTypeObject* tp = Py_TYPE(op);
	delete self->e;
	self->e = NULL;
	delete self->mu;
	self->mu = NULL;
	tp->tp_free(op);
	Py_DECREF(tp);
}

static PyType_Slot bk_slots[] = {
	{Py_tp_new, (void*)PyType_GenericNew},
	{Py_tp_init, (void*)bk_init},
	{Py_tp_dealloc, (void*)bk_dealloc},
	{Py_tp_methods, (void*)bk_methods},
	{Py_tp_members, (void*)bk_members},
	{Py_tp_doc, (void*)"Low-level handle to a Kolmogorov-Zabih Energy instance."},
	{0, NULL}
};

static PyType_Spec bk_spec = {
	"bkgco._bkgco.BKEnergy",
	sizeof(BKObj),
	0,
	Py_TPFLAGS_DEFAULT,
	bk_slots
};

/////////////////////////////////////////////////////////////////////
// BKGraph: the basic interface of Graph<> (graph.h), fed in batches.
// Nodes are created once, up front; the advanced interface (tree reuse,
// residual-capacity access, reset, arc iteration) is not exposed.

struct GraphBackend {
	virtual ~GraphBackend() { }
	virtual void addTweights(const void* caps, Py_ssize_t n) = 0;
	virtual void addEdges(const int* idx, const void* caps, Py_ssize_t n) = 0;
	virtual EnergyVal maxflow() = 0;
	virtual void getSegments(unsigned char* out, int n, int deflt) = 0;
};

template <typename Term>
struct GraphBackendT: GraphBackend {
	typedef Graph<Term, Term, typename GCO_Traits<Term>::TotalType> G;

	GraphBackendT(int numNodes, int edgeCapacity): m_g(new G(numNodes, edgeCapacity, bk_error))
	{
		m_g->add_node(numNodes);
	}
	~GraphBackendT() { delete m_g; }

	void addTweights(const void* caps, Py_ssize_t n)
	{
		const Term* c = (const Term*)caps;
		for (Py_ssize_t i = 0; i < n; ++i) {
			check(c[2*i], i);       // terminal capacities may be negative
			check(c[2*i+1], i);
			m_g->add_tweights((int)i, c[2*i], c[2*i+1]);
		}
	}

	void addEdges(const int* idx, const void* caps, Py_ssize_t n)
	{
		const Term* c = (const Term*)caps;
		for (Py_ssize_t i = 0; i < n; ++i) {
			check(c[2*i], i);
			check(c[2*i+1], i);
			if (c[2*i] < 0 || c[2*i+1] < 0)
				throw BKError(bk_msg(i, "edge capacities must be non-negative"));
			m_g->add_edge(idx[2*i], idx[2*i+1], c[2*i], c[2*i+1]);
		}
	}

	EnergyVal maxflow()
	{
		EnergyVal v;
		v.isf = std::is_same<Term, double>::value;
		typename GCO_Traits<Term>::TotalType f = m_g->maxflow();
		v.i = (long long)f;
		v.d = (double)f;
		return v;
	}

	void getSegments(unsigned char* out, int n, int deflt)
	{
		typename G::termtype d = deflt ? G::SINK : G::SOURCE;
		for (int i = 0; i < n; ++i)
			out[i] = (unsigned char)m_g->what_segment(i, d);
	}

private:
	static void check(Term v, Py_ssize_t row)
	{
		Term m = GCO_Traits<Term>::maxTerm();
		if (v > m || v < -m)
			throw BKError(bk_msg(row, "capacity magnitude exceeds max_energy_term(dtype)"));
	}
	G* m_g;
};

struct BKGObj {
	PyObject_HEAD
	GraphBackend* g;
	std::mutex* mu;
	int dtype_code;
	int is_float;
	int nnodes;
	int flowed;
};

static const char* bkg_term_fmt(BKGObj* self)
{
	return self->dtype_code == GCO_INT32 ? "il" : (self->dtype_code == GCO_INT64 ? "lq" : "d");
}
static Py_ssize_t bkg_term_size(BKGObj* self) { return self->dtype_code == GCO_INT32 ? 4 : 8; }

static int bkg_check_open(BKGObj* self)
{
	if (!self->g) {
		PyErr_SetString(PyExc_RuntimeError, "BKGraph object has been destroyed");
		return 0;
	}
	return 1;
}

static PyObject* bkg_add_tweights(BKGObj* self, PyObject* arg)
{
	if (!bkg_check_open(self))
		return NULL;
	Py_buffer b;
	if (getbuf(arg, &b, 0, bkg_term_fmt(self), bkg_term_size(self),
	           (Py_ssize_t)self->nnodes * 2, "terminal capacities") < 0)
		return NULL;
	try {
		Unlocked u(self->mu);
		self->g->addTweights(b.buf, self->nnodes);
	} catch (BKError& x) {
		PyBuffer_Release(&b);
		PyErr_SetString(PyExc_ValueError, x.msg.c_str());
		return NULL;
	} catch (GCException& x) {
		PyBuffer_Release(&b);
		return raise_gc(x.message);
	} catch (...) {
		PyBuffer_Release(&b);
		return raise_gc("unknown error inside Graph");
	}
	PyBuffer_Release(&b);
	Py_RETURN_NONE;
}

static PyObject* bkg_add_edges(BKGObj* self, PyObject* args)
{
	PyObject* idx_o;
	PyObject* cap_o;
	if (!PyArg_ParseTuple(args, "OO", &idx_o, &cap_o))
		return NULL;
	if (!bkg_check_open(self))
		return NULL;
	Py_buffer ib, cb;
	Py_ssize_t n = getbuf(idx_o, &ib, 0, kIdxFmt, 4, -1, "edge nodes");
	if (n < 0)
		return NULL;
	if (n % 2) {
		PyBuffer_Release(&ib);
		PyErr_SetString(PyExc_ValueError, "edge nodes: expected 2 columns");
		return NULL;
	}
	n /= 2;
	const int* idx = (const int*)ib.buf;
	for (Py_ssize_t i = 0; i < n; ++i) {
		if (idx[2*i] < 0 || idx[2*i] >= self->nnodes || idx[2*i+1] < 0 || idx[2*i+1] >= self->nnodes) {
			PyBuffer_Release(&ib);
			PyErr_Format(PyExc_ValueError, "row %zd: node index out of range 0..%d", i, self->nnodes - 1);
			return NULL;
		}
		if (idx[2*i] == idx[2*i+1]) {
			PyBuffer_Release(&ib);
			PyErr_Format(PyExc_ValueError, "row %zd: an edge needs two distinct nodes", i);
			return NULL;
		}
	}
	if (getbuf(cap_o, &cb, 0, bkg_term_fmt(self), bkg_term_size(self), n * 2, "edge capacities") < 0) {
		PyBuffer_Release(&ib);
		return NULL;
	}
	PyObject* res = NULL;
	try {
		Unlocked u(self->mu);
		self->g->addEdges(idx, cb.buf, n);
		res = Py_None;
	} catch (BKError& x) {
		PyErr_SetString(PyExc_ValueError, x.msg.c_str());
	} catch (GCException& x) {
		raise_gc(x.message);
	} catch (std::bad_alloc&) {
		PyErr_NoMemory();
	} catch (...) {
		raise_gc("unknown error inside Graph");
	}
	PyBuffer_Release(&ib);
	PyBuffer_Release(&cb);
	return res ? Py_NewRef(res) : NULL;
}

static PyObject* bkg_maxflow(BKGObj* self, PyObject* Py_UNUSED(a))
{
	if (!bkg_check_open(self))
		return NULL;
	EnergyVal v;
	try {
		Unlocked u(self->mu);
		v = self->g->maxflow();
	}
	BK_CATCH
	self->flowed = 1;
	return v.toPy();
}

static PyObject* bkg_get_segments(BKGObj* self, PyObject* args)
{
	PyObject* out_o;
	int deflt = 0;
	if (!PyArg_ParseTuple(args, "O|i", &out_o, &deflt))
		return NULL;
	if (!bkg_check_open(self))
		return NULL;
	if (!self->flowed) {
		PyErr_SetString(PyExc_RuntimeError, "call maxflow() before get_segments()");
		return NULL;
	}
	if (deflt != 0 && deflt != 1) {
		PyErr_SetString(PyExc_ValueError, "default segment must be 0 (source) or 1 (sink)");
		return NULL;
	}
	Py_buffer b;
	if (getbuf(out_o, &b, 1, "B", 1, self->nnodes, "segment output") < 0)
		return NULL;
	{
		Unlocked u(self->mu);
		self->g->getSegments((unsigned char*)b.buf, self->nnodes, deflt);
	}
	PyBuffer_Release(&b);
	Py_RETURN_NONE;
}

static PyObject* bkg_destroy(BKGObj* self, PyObject* Py_UNUSED(a))
{
	if (self->g) {
		Unlocked u(self->mu);
		delete self->g;
		self->g = NULL;
	}
	Py_RETURN_NONE;
}

static PyMethodDef bkg_methods[] = {
	{"add_tweights", (PyCFunction)bkg_add_tweights, METH_O, NULL},
	{"add_edges", (PyCFunction)bkg_add_edges, METH_VARARGS, NULL},
	{"maxflow", (PyCFunction)bkg_maxflow, METH_NOARGS, NULL},
	{"get_segments", (PyCFunction)bkg_get_segments, METH_VARARGS, NULL},
	{"destroy", (PyCFunction)bkg_destroy, METH_NOARGS, NULL},
	{NULL, NULL, 0, NULL}
};

static PyMemberDef bkg_members[] = {
	{"num_nodes", Py_T_INT, offsetof(BKGObj, nnodes), Py_READONLY, NULL},
	{"dtype_code", Py_T_INT, offsetof(BKGObj, dtype_code), Py_READONLY, NULL},
	{"is_float64", Py_T_INT, offsetof(BKGObj, is_float), Py_READONLY, NULL},
	{"flowed", Py_T_INT, offsetof(BKGObj, flowed), Py_READONLY, NULL},
	{NULL, 0, 0, 0, NULL}
};

static int bkg_init(PyObject* op, PyObject* args, PyObject* kwds)
{
	static const char* kwlist[] = {"num_nodes", "dtype_code", "edge_capacity", NULL};
	BKGObj* self = (BKGObj*)op;
	int nnodes = 0, code = GCO_INT32, edges = 0;
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "i|ii", (char**)kwlist, &nnodes, &code, &edges))
		return -1;
	if (self->g) {
		PyErr_SetString(PyExc_RuntimeError, "BKGraph object is already initialized");
		return -1;
	}
	if (nnodes < 1) {
		PyErr_SetString(PyExc_ValueError, "num_nodes must be >= 1");
		return -1;
	}
	if (edges < 0) {
		PyErr_SetString(PyExc_ValueError, "edge_capacity must be >= 0");
		return -1;
	}
	if (code < GCO_INT32 || code > GCO_FLOAT64) {
		PyErr_SetString(PyExc_ValueError, "dtype_code must be 0 (int32), 1 (int64) or 2 (float64)");
		return -1;
	}
	try {
		self->mu = new std::mutex();
		switch (code) {
		case GCO_INT64:   self->g = new GraphBackendT<long long>(nnodes, edges); break;
		case GCO_FLOAT64: self->g = new GraphBackendT<double>(nnodes, edges); break;
		default:          self->g = new GraphBackendT<int>(nnodes, edges); break;
		}
	} catch (GCException& x) {
		raise_gc(x.message);
		return -1;
	} catch (std::bad_alloc&) {
		PyErr_NoMemory();
		return -1;
	}
	self->nnodes = nnodes;
	self->dtype_code = code;
	self->is_float = (code == GCO_FLOAT64) ? 1 : 0;
	self->flowed = 0;
	return 0;
}

static void bkg_dealloc(PyObject* op)
{
	BKGObj* self = (BKGObj*)op;
	PyTypeObject* tp = Py_TYPE(op);
	delete self->g;
	self->g = NULL;
	delete self->mu;
	self->mu = NULL;
	tp->tp_free(op);
	Py_DECREF(tp);
}

static PyType_Slot bkg_slots[] = {
	{Py_tp_new, (void*)PyType_GenericNew},
	{Py_tp_init, (void*)bkg_init},
	{Py_tp_dealloc, (void*)bkg_dealloc},
	{Py_tp_methods, (void*)bkg_methods},
	{Py_tp_members, (void*)bkg_members},
	{Py_tp_doc, (void*)"Low-level handle to a Boykov-Kolmogorov maxflow Graph instance."},
	{0, NULL}
};

static PyType_Spec bkg_spec = {
	"bkgco._bkgco.BKGraph",
	sizeof(BKGObj),
	0,
	Py_TPFLAGS_DEFAULT,
	bkg_slots
};

struct ModState {
	PyObject* type;
	PyObject* bk_type;
	PyObject* bkg_type;
};

static int mod_exec(PyObject* m)
{
	ModState* st = (ModState*)PyModule_GetState(m);
	st->type = PyType_FromModuleAndSpec(m, &gco_spec, NULL);
	if (!st->type)
		return -1;
	if (PyModule_AddObjectRef(m, "GCO", st->type) < 0)
		return -1;
	st->bk_type = PyType_FromModuleAndSpec(m, &bk_spec, NULL);
	if (!st->bk_type)
		return -1;
	if (PyModule_AddObjectRef(m, "BKEnergy", st->bk_type) < 0)
		return -1;
	st->bkg_type = PyType_FromModuleAndSpec(m, &bkg_spec, NULL);
	if (!st->bkg_type)
		return -1;
	if (PyModule_AddObjectRef(m, "BKGraph", st->bkg_type) < 0)
		return -1;
	// largest accepted term per dtype code, in the same order as the codes
	PyObject* caps = Py_BuildValue("(iLd)", GCoptimizationT<int>::maxEnergyTerm(),
	                               (long long)GCoptimizationT<long long>::maxEnergyTerm(),
	                               GCoptimizationT<double>::maxEnergyTerm());
	if (!caps || PyModule_AddObject(m, "max_energy_terms", caps) < 0) {
		Py_XDECREF(caps);
		return -1;
	}
	return 0;
}

static int mod_traverse(PyObject* m, visitproc visit, void* arg)
{
	ModState* st = (ModState*)PyModule_GetState(m);
	Py_VISIT(st->type);
	Py_VISIT(st->bk_type);
	Py_VISIT(st->bkg_type);
	return 0;
}

static int mod_clear(PyObject* m)
{
	ModState* st = (ModState*)PyModule_GetState(m);
	Py_CLEAR(st->type);
	Py_CLEAR(st->bk_type);
	Py_CLEAR(st->bkg_type);
	return 0;
}

static PyModuleDef_Slot mod_slots[] = {
	{Py_mod_exec, (void*)mod_exec},
	{Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
	{Py_mod_gil, Py_MOD_GIL_NOT_USED},
	{0, NULL}
};

static PyModuleDef gco_module = {
	PyModuleDef_HEAD_INIT,
	"_bkgco",
	"GCoptimization, Energy and maxflow Graph bindings (int32, int64, float64).",
	sizeof(ModState),
	NULL,
	mod_slots,
	mod_traverse,
	mod_clear,
	NULL
};

PyMODINIT_FUNC PyInit__bkgco(void)
{
	return PyModuleDef_Init(&gco_module);
}
