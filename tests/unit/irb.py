"""Terse constructors for hand-built zuspec IR in tests."""
import zuspec.ir.core as ir


def u(w):
    return ir.DataTypeInt(bits=w, signed=False)


def s(w):
    return ir.DataTypeInt(bits=w, signed=True)


def arr(elem, n):
    return ir.DataTypeArray(element_type=elem, size=n)


def c(v, dt=None):
    """A constant; with *dt*, wrapped in a cast to that type."""
    k = ir.ExprConstant(value=v)
    return k if dt is None else ir.ExprCast(target_type=dt, value=k)


def cast(dt, e):
    return ir.ExprCast(target_type=dt, value=e)


def p(name):
    return ir.ExprRefParam(name=name)


def l(name):
    return ir.ExprRefLocal(name=name)


def fld(i):
    return ir.ExprRefField(base=ir.TypeExprRefSelf(), index=i)


def bop(a, op, b):
    return ir.ExprBin(lhs=a, op=getattr(ir.BinOp, op), rhs=b)


def cmp(a, op, b):
    return ir.ExprCompare(left=a, ops=[getattr(ir.CmpOp, op)], comparators=[b])


def un(op, a):
    return ir.ExprUnary(op=getattr(ir.UnaryOp, op), operand=a)


def ifx(t, a, b):
    return ir.ExprIfExp(test=t, body=a, orelse=b)


def idx(a, i):
    return ir.ExprSubscript(value=a, slice=i)


def ps(v, base, w):
    return ir.ExprPartSelect(value=v, base=base, width=w)


def cat(*vs):
    return ir.ExprConcat(values=list(vs))


def call(name, *args):
    return ir.ExprCall(func=ir.ExprAttribute(value=ir.TypeExprRefSelf(), attr=name),
                       args=list(args))


def assign(t, v):
    return ir.StmtAssign(targets=[t], value=v)


def decl(name, dt, v=None):
    return ir.StmtAnnAssign(target=l(name), annotation=ir.ExprConstant(value=None),
                            ir_type=dt, value=v)


def if_(t, body, orelse=()):
    return ir.StmtIf(test=t, body=list(body), orelse=list(orelse))


def ret(v):
    return ir.StmtReturn(value=v)


def for_(var, n, body, dt=None):
    """``for (int var = 0; var < n; var++) body``, as the front end writes it."""
    dt = dt or s(32)
    return [decl(var, dt, c(0, dt)),
            ir.StmtWhile(test=cmp(l(var), "Lt", c(n, dt)),
                         body=list(body) + [assign(l(var), bop(l(var), "Add", c(1, dt)))])]


def fn(name, args, returns, body):
    return ir.Function(
        name=name,
        args=ir.Arguments(args=[ir.Arg(arg=a, datatype=dt) for a, dt in args]),
        returns=returns, body=list(body))


def get(port_idx):
    return ir.ExprAwait(value=ir.ExprCall(
        func=ir.ExprAttribute(value=fld(port_idx), attr="get"), args=[]))


def put(port_idx, v):
    return ir.StmtExpr(expr=ir.ExprAwait(value=ir.ExprCall(
        func=ir.ExprAttribute(value=fld(port_idx), attr="put"), args=[v])))


def forever(*body):
    return ir.StmtWhile(test=ir.ExprConstant(value=True), body=list(body))


def component(name, fields, run_body, functions=()):
    comp = ir.DataTypeComponent(name=name, super=None, fields=list(fields),
                                functions=list(functions))
    comp.proc_processes = [ir.Function(name="run", body=list(run_body), is_async=True)]
    return comp


def field(name, dt, init=None):
    return ir.Field(name=name, datatype=dt, initial_value=init)


def in_port(name, dt):
    return ir.Field(name=name, datatype=ir.DataTypeGetIF(element_type=dt), kind=ir.FieldKind.Port)


def out_port(name, dt):
    return ir.Field(name=name, datatype=ir.DataTypePutIF(element_type=dt), kind=ir.FieldKind.Port)
